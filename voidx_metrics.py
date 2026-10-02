"""
voidx_metrics.py
================
Evaluation metrics for the VoidX pipeline (Section VI-B of the paper).

Metrics implemented (all normalised by mesh bounding-box diagonal
unless stated otherwise):
    - chamfer_distance_j2j   (CD-J2J)   symmetric joint-to-joint
    - chamfer_distance_j2b   (CD-J2B)   symmetric joint-to-bone
    - chamfer_distance_b2b   (CD-B2B)   symmetric bone-to-bone
    - PCK (IoU/Precision/Recall)        Hungarian-matched correspondence;
                                        a predicted joint is correct if
                                        within tau x bbox diagonal of its
                                        matched reference joint (tau=5%)
    - tree_edit_distance                skeleton-tree edit distance
    - volumetric_centering_error (VCE)  radial distance to the local
                                        cross-section centre, normalised
                                        by local diameter

Metric definitions follow the RigNet paper (Xu et al., SIGGRAPH 2020)
for the Chamfer measures, but (i) evaluation is restricted to the 14
core joints present in both prediction and reference, and (ii) the
correspondence criterion is PCK with an absolute tolerance, because
reference pivots follow animator conventions and lie at or near the
mesh surface (median 1.34% of diagonal), so reference-derived
tolerances measure placement convention rather than prediction error.
Results are therefore NOT directly comparable to full-skeleton
protocols."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# =====================================================================
# Skeleton / bone representation
# =====================================================================
BONE_CONNECTIONS: List[Tuple[str, str]] = [
    ("pelvis", "hip_L"), ("pelvis", "hip_R"),
    ("hip_L", "knee_L"), ("hip_R", "knee_R"),
    ("knee_L", "ankle_L"), ("knee_R", "ankle_R"),
    ("pelvis", "neck"),
    ("neck", "shoulder_L"), ("neck", "shoulder_R"),
    ("shoulder_L", "elbow_L"), ("shoulder_R", "elbow_R"),
    ("elbow_L", "wrist_L"), ("elbow_R", "wrist_R"),
]

def chamfer_distance_j2j_matched(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
    mesh,
) -> float:
    """Matched CD-J2J: For every predicted joint, find the closest GT 
    joint and calculate distance only between matched pairs. 
    This prevents penalizing the pipeline for not predicting 40+ 
    auxiliary joints (fingers, spine) that MediaPipe doesn't track."""
    P = joints_to_array(pred)
    G = joints_to_array(gt)
    if len(P) == 0 or len(G) == 0:
        return float("inf")
    
    # Find the closest GT joint for each of our predicted joints
    tree_g = cKDTree(G)
    distances, indices = tree_g.query(P, k=1)
    
    # Get the matched GT joints (should be 14 pairs)
    matched_G = G[indices]
    
    # Calculate symmetric Chamfer distance on matched pairs only
    tree_mg = cKDTree(matched_G)
    d_pg, _ = tree_mg.query(P, k=1)
    tree_p = cKDTree(P)
    d_gp, _ = tree_p.query(matched_G, k=1)
    
    cd = 0.5 * (float(np.mean(d_pg)) + float(np.mean(d_gp)))
    diag = bbox_diagonal(mesh)
    return cd / diag * 100.0

def matched_iou_precision_recall(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
    mesh,
    tolerance_fraction: float = 0.5,
) -> Tuple[float, float, float]:
    """Matched IoU / Precision / Recall using nearest-neighbour matching
    with tolerance = 1/2 local shape diameter.

    ── FIX #3: Standard IR formulas ──
        precision = TP / (TP + FP) = matched / |P|
        recall    = TP / (TP + FN) = matched / |G|
        IoU       = TP / (TP + FP + FN) = matched / (|P| + |G| - matched)

    The previous version returned iou=precision and recall=precision,
    which made every IoU and Recall number in the results CSV wrong.
    """
    P = joints_to_array(pred)
    G = joints_to_array(gt)
    if len(P) == 0 or len(G) == 0:
        return 0.0, 0.0, 0.0

    # Find closest GT joint for each prediction
    tree_g = cKDTree(G)
    distances, indices = tree_g.query(P, k=1)

    matched = 0
    for i, dist in enumerate(distances):
        gt_point = G[indices[i]]
        tol = tolerance_fraction * _shape_diameter(mesh, gt_point)
        if dist <= tol:
            matched += 1

    # ── FIX #3: correct formulas ──
    precision = matched / len(P) * 100.0                          
    recall    = matched / len(G) * 100.0                         
    union     = len(P) + len(G) - matched
    iou       = (matched / union * 100.0) if union > 0 else 0.0  

    return iou, precision, recall

def joints_to_array(skeleton: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Convert a skeleton dict to an (N, 3) array, preserving the order
    of ``BONE_CONNECTIONS``-relevant joints."""
    return np.array([
        [skeleton[j]["x"], skeleton[j]["y"], skeleton[j]["z"]]
        for j in skeleton
        if isinstance(skeleton[j], dict) and "x" in skeleton[j]
    ])


def bones_to_segments(
    skeleton: Dict[str, Dict[str, float]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract bone line segments from a skeleton dict using the
    canonical BONE_CONNECTIONS topology."""
    segs = []
    for a, b in BONE_CONNECTIONS:
        if a in skeleton and b in skeleton:
            pa = np.array([skeleton[a]["x"], skeleton[a]["y"], skeleton[a]["z"]])
            pb = np.array([skeleton[b]["x"], skeleton[b]["y"], skeleton[b]["z"]])
            segs.append((pa, pb))
    return segs


def bbox_diagonal(mesh) -> float:
    """Mesh bounding-box diagonal — used to normalise Chamfer distances
    so they are scale-invariant."""
    b_min = np.asarray(mesh.bounds[0])
    b_max = np.asarray(mesh.bounds[1])
    return float(np.linalg.norm(b_max - b_min))


# =====================================================================
# CD-J2J: symmetric Chamfer distance joint-to-joint
# =====================================================================
def chamfer_distance_j2j(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
    mesh,
) -> float:
    """Symmetric joint-to-joint Chamfer distance, normalised by the mesh
    bounding-box diagonal and reported as a percentage.

        CD-J2J = 0.5 * (mean_i min_j ||p_i - g_j||  +
                        mean_j min_i ||g_j - p_i||) / diag * 100
    """
    P = joints_to_array(pred)
    G = joints_to_array(gt)
    if len(P) == 0 or len(G) == 0:
        return float("inf")
    tree_g = cKDTree(G)
    tree_p = cKDTree(P)
    d_pg, _ = tree_g.query(P, k=1)
    d_gp, _ = tree_p.query(G, k=1)
    cd = 0.5 * (float(np.mean(d_pg)) + float(np.mean(d_gp)))
    diag = bbox_diagonal(mesh)
    return cd / diag * 100.0


# =====================================================================
# CD-J2B: joint-to-bone Chamfer distance
# =====================================================================
def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point ``p`` to segment ``[a, b]``."""
    ab = b - a
    ap = p - a
    t = float(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12))
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def chamfer_distance_j2b(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
    mesh,
) -> float:
    """Symmetric joint-to-bone Chamfer distance, normalised."""
    P = joints_to_array(pred)
    G = joints_to_array(gt)
    pred_bones = bones_to_segments(pred)
    gt_bones = bones_to_segments(gt)
    diag = bbox_diagonal(mesh)

    def mean_dist_to_bones(joints, bones):
        if not bones or len(joints) == 0:
            return 0.0
        ds = []
        for j in joints:
            best = min(_point_to_segment_dist(j, a, b) for a, b in bones)
            ds.append(best)
        return float(np.mean(ds))

    def mean_dist_to_joints(bones, joints):
        if not bones or len(joints) == 0:
            return 0.0
        tree = cKDTree(joints)
        ds = []
        for a, b in bones:
            # sample 5 points along the bone
            for t in np.linspace(0, 1, 5):
                p = a + t * (b - a)
                d, _ = tree.query(p, k=1)
                ds.append(float(d))
        return float(np.mean(ds))

    # FIX M3: symmetric J2B = pred joints→GT bones + GT joints→pred bones
    cd = 0.5 * (mean_dist_to_bones(P, gt_bones) +
                mean_dist_to_bones(G, pred_bones))    
    return cd / diag * 100.0


# =====================================================================
# CD-B2B: bone-to-bone Chamfer distance
# =====================================================================
def chamfer_distance_b2b(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
    mesh,
) -> float:
    """Symmetric bone-to-bone Chamfer distance, normalised."""
    pred_bones = bones_to_segments(pred)
    gt_bones = bones_to_segments(gt)
    diag = bbox_diagonal(mesh)

    def mean_bone_to_bones(A, B):
        if not A or not B:
            return 0.0
        ds = []
        for a1, a2 in A:
            for t in np.linspace(0, 1, 5):
                p = a1 + t * (a2 - a1)
                best = min(_point_to_segment_dist(p, b1, b2) for b1, b2 in B)
                ds.append(best)
        return float(np.mean(ds))

    cd = 0.5 * (mean_bone_to_bones(pred_bones, gt_bones) +
                mean_bone_to_bones(gt_bones, pred_bones))
    return cd / diag * 100.0


# =====================================================================
# IoU + Precision + Recall via Hungarian matching
# =====================================================================
def _shape_diameter(mesh, point: np.ndarray, n_rays: int = 16) -> float:
    """Local shape diameter. Uses a robust fallback for non-watertight AI meshes."""
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    try:
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(n_rays, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        origins = np.tile(point, (n_rays, 1))
        locs, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs
        )
        if len(locs) == 0:
            # Fallback for AI meshes with holes: use 5% of bounding box
            return diag * 0.05 
            
        lengths = []
        for i in range(n_rays):
            hits = locs[idx_ray == i]
            if len(hits) > 0:
                lengths.append(float(np.linalg.norm(hits[0] - point)))
                
        if lengths:
            # FIX M1: median, not max. Max picks up rays traveling ALONG
            # the limb (length ≈ limb length), inflating the tolerance.
            # Median ≈ local cross-sectional thickness.
            return float(np.median(lengths))
        
    except Exception:
        # Fallback if raycasting crashes on messy geometry
        return diag * 0.05


def joint_iou_precision_recall(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
    mesh,
    tolerance_fraction: float = 0.05,
) -> Tuple[float, float, float]:
    """PCK-style correspondence (final calibration): a predicted joint is
    correct if it lies within `tolerance_fraction` x bbox diagonal of its
    Hungarian-matched reference joint — the standard criterion from human
    pose estimation.

    Chosen over reference-derived tolerances (surface distance, ray-cast
    diameter) because reference pivots lie at or near the mesh surface
    (median 1.34% of diagonal), making those tolerances measure pivot
    placement convention, not prediction quality. The full tau curve
    (2.5/5/7.5/10%) is reported from the frozen-prediction calibration.
    """
    P = joints_to_array(pred)
    G = joints_to_array(gt)
    if len(P) == 0 or len(G) == 0:
        return 0.0, 0.0, 0.0

    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    tol = tolerance_fraction * bbox_diagonal(mesh)
    matched = int(np.sum(cost[row_ind, col_ind] <= tol))

    precision = matched / len(P) * 100.0
    recall = matched / len(G) * 100.0
    union = len(P) + len(G) - matched
    iou = (matched / union * 100.0) if union > 0 else 0.0
    return iou, precision, recall


# =====================================================================
# Tree Edit Distance
# =====================================================================
def tree_edit_distance(
    pred: Dict[str, Dict[str, float]],
    gt: Dict[str, Dict[str, float]],
) -> int:
    """Minimum number of joint insertions/deletions to transform the
    predicted skeleton tree into the reference tree.

    Approximate via the symmetric difference of joint sets weighted by
    topology differences (a simplified Zhang-Shasha-style metric).
    """
    pred_names = set(pred.keys())
    gt_names = set(gt.keys())
    # Joints to delete from pred (in pred but not in gt) + joints to insert (in gt but not in pred)
    edits = len(pred_names - gt_names) + len(gt_names - pred_names)
    # Add 1 edit for each bone that exists in only one skeleton
    pred_bones = {(a, b) for a, b in BONE_CONNECTIONS
                  if a in pred_names and b in pred_names}
    gt_bones = {(a, b) for a, b in BONE_CONNECTIONS
                if a in gt_names and b in gt_names}
    edits += len(pred_bones ^ gt_bones)
    return edits


# =====================================================================
# VCE: Volumetric Centering Error (paper's new metric)
# =====================================================================
def volumetric_centering_error(
    pred: Dict[str, Dict[str, float]],
    mesh,
) -> float:
    """Mean Euclidean distance from each predicted joint to the medial
    axis of the nearest limb segment, normalised by the local shape
    diameter.

    This is the new metric introduced in Section VI-B of the paper to
    quantify how well joints are centred inside the mesh volume (the
    primary contribution of the pipeline).

    For each joint, we approximate the medial axis as the locus of
    maximal inscribed spheres along the nearest bone segment, computed
    by firing 8 radial rays and taking the centre of the minimum
    inscribed disc.
    """
    bones = bones_to_segments(pred)
    if not bones:
        return float("inf")

    errors = []
    for joint_name, pos in pred.items():
        if not isinstance(pos, dict) or "x" not in pos:
            continue
        p = np.array([pos["x"], pos["y"], pos["z"]])

        # Find nearest bone segment
        best_seg = None
        best_d = float("inf")
        for a, b in bones:
            d = _point_to_segment_dist(p, a, b)
            if d < best_d:
                best_d = d
                best_seg = (a, b)
        if best_seg is None:
            continue

        # Sample the medial axis along this bone: fire 8 radial rays at
        # the projection of p onto the bone, take the minimum hit radius
        a, b = best_seg
        ab = b - a
        t = float(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12))
        t = max(0.0, min(1.0, t))
        proj = a + t * ab

        # Sample 8 directions in the plane perpendicular to the bone
        bone_dir = ab / max(np.linalg.norm(ab), 1e-12)
        # Build an orthonormal basis (u, v) perpendicular to bone_dir
        ref = np.array([1.0, 0.0, 0.0]) if abs(bone_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(bone_dir, ref)
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(bone_dir, u)

        radii = []
        hit_by_dir = {}   # FIX #7: track hit per direction index
        for k in range(8):
            ang = 2 * math.pi * k / 8
            direction = math.cos(ang) * u + math.sin(ang) * v
            try:
                locs, idx_ray, _ = mesh.ray.intersects_location(
                    ray_origins=np.array([proj]),
                    ray_directions=np.array([direction]),
                )
                if len(locs) > 0:
                    radii.append(float(np.linalg.norm(locs[0] - proj)))
                    hit_by_dir[k] = np.asarray(locs[0])
            except Exception:
                pass

        if not radii:
            continue

        # ── FIX #7: center from OPPOSITE-PAIR midpoints ──
        # k and k+4 are antipodal; midpoint of their hits = unbiased
        # center estimate along that axis, even for asymmetric sections.
        midpoints, pair_lengths = [], []
        for k in range(4):
            if k in hit_by_dir and (k + 4) in hit_by_dir:
                pa, pb = hit_by_dir[k], hit_by_dir[k + 4]
                midpoints.append(0.5 * (pa + pb))
                pair_lengths.append(float(np.linalg.norm(pa - pb)))
        if len(midpoints) < 2:
            continue

        center = np.mean(midpoints, axis=0)
        local_diameter = float(np.median(pair_lengths))
        if local_diameter < 1e-6:
            continue

        # Radial error only: strip the along-bone component
        err_vec = p - center
        radial = err_vec - np.dot(err_vec, bone_dir) * bone_dir
        radial_err = float(np.linalg.norm(radial))
        errors.append(radial_err / local_diameter)

    # ── after the loop ──
    if not errors:
        # Unmeasurable → inf → _finite() filters it → excluded from the
        # mean, and VCE_n tells you HOW MANY were excluded.
        return float("inf")
    return float(np.mean(errors))


# =====================================================================
# Convenience: compute all metrics at once
# =====================================================================
def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def compute_all_metrics(pred, gt, mesh):
    iou, prec, rec = joint_iou_precision_recall(pred, gt, mesh)   # FIX M2: Hungarian
    vce = volumetric_centering_error(pred, mesh)
    def _r(x):
        return round(x, 4) if _finite(x) else None
    return {
        "CD-J2J": _r(chamfer_distance_j2j(pred, gt, mesh)),        
        "CD-J2B": _r(chamfer_distance_j2b(pred, gt, mesh)),       
        "CD-B2B": _r(chamfer_distance_b2b(pred, gt, mesh)),       
        "IoU": round(iou, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "ED": tree_edit_distance(pred, gt),
        "VCE": _r(vce),
    }