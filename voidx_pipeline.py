"""
voidx_pipeline.py
=================
End-to-end implementation of the deterministic volumetric auto-rigging
pipeline described in Algorithm 1 of:

    "Deterministic Volumetric Auto-Rigging via Multi-View Orthographic
     Lifting of 2D Pose Landmarks" (Krishnanand, 2026)

Pipeline phases (paper Section IV):
    Phase 1 - Multi-View Spatial Extraction
        1.1  MediaPipe conditioning + orthographic camera setup (Sec. IV-A)
        1.2  Semantic Jurisdiction Zones filtering        (Sec. IV-B, Table I)
        1.3  Decoupled Axis Projection                     (Sec. IV-C, Thm 4)
        1.4  Kinematic Inheritance for occluded joints     (Sec. IV-D, Table II)
        1.5  Cranial offset + dynamic scaling              (Sec. IV-E, Eqs. 3-4)
    Phase 2 - Volumetric Master Engine
        2.1  Proximity Engine (surface initialisation)     (Sec. IV-F.1)
        2.2  Y-axis volumetric centering (5-ray micro-grid) (Sec. IV-F.2, Eq. 5)
        2.3  Z-axis micro-grid + anatomical thickness caps (Sec. IV-F.3, Eq. 6,
                                                          Prop. 8)

NOTE on the depth estimator:
    Theorem 4 proves LS is optimal under i.i.d. Gaussian noise. However,
    MediaPipe detection noise is heteroscedastic (side views have much
    higher variance and systematic bias than frontal views). Empirically,
    the single-angle inversion (Theorem 5) — which uses only the cleanest
    views for each axis — outperforms LS on real MediaPipe observations.
    Therefore the DEFAULT is use_ls_estimator=False. The LS estimator is
    retained for the ablation study (Table VIII) and for environments
    with synthetic i.i.d. noise (e.g., Monte-Carlo verification).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from voidx_theorems import (
    AnthroCaps,
    corollary_1_y_decouple,
    proposition_7_midpoint,
    proposition_8_anthro_caps,
    thickness_cap_for_joint,
    theorem_1_projection,
    theorem_2_min_views,
    theorem_3_crlb,
    theorem_4_ls_optimal,
    theorem_5_singularity,
)

def load_mesh_trimesh(mesh_path: str):
    """FIX #12: single shared mesh loader for the pipeline AND the
    experiment runner. Handles multi-object OBJs/GLBs that come back
    as a Scene with node transforms."""
    import trimesh
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        geoms = []
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph.get(node)
            geom = scene.geometry[geometry_name].copy()   # .copy() = don't corrupt the cache
            geom.apply_transform(transform)
            geoms.append(geom)
        return trimesh.util.concatenate(geoms)
    return scene
# =====================================================================
# Configuration (Paper Section IV-A)
# =====================================================================
@dataclass
class PipelineConfig:
    """Hyperparameters of the pipeline.  Defaults match the values stated
    in Section IV-A and IV-F of the paper."""
    # MediaPipe conditioning (Sec. IV-A)
    model_complexity: int = 2
    min_detection_confidence: float = 0.5
    mesh_skin_tone: Tuple[float, float, float] = (0.9, 0.8, 0.7)
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Orthographic camera (Sec. IV-A)
    orthographic_scale_factor: float = 20.0  # Eq.: focal *= 20, translation *= 20
    render_resolution: int = 1024

    # Multi-view scanning (Sec. IV-A)
    scan_angles_deg: Sequence[int] = (0, 45, 90, 135, 180, 225, 270, 315)

    # Skeleton: 14 core joints (Sec. III-A)
    core_joints: Sequence[str] = (
        "shoulder_L", "shoulder_R",
        "elbow_L", "elbow_R",
        "wrist_L", "wrist_R",
        "hip_L", "hip_R",
        "knee_L", "knee_R",
        "ankle_L", "ankle_R",
    )

    # MediaPipe landmark indices (BlazePose 33-landmark scheme)
    mediapipe_indices: Dict[str, int] = field(default_factory=lambda: {
        "shoulder_L": 11, "shoulder_R": 12,
        "elbow_L": 13, "elbow_R": 14,
        "wrist_L": 15, "wrist_R": 16,
        "hip_L": 23, "hip_R": 24,
        "knee_L": 25, "knee_R": 26,
        "ankle_L": 27, "ankle_R": 28,
    })

    # Semantic Jurisdiction Zones (Paper Table I)
    jurisdiction_zones: Dict[str, Sequence[int]] = field(default_factory=lambda: {
        # Front-Facing Core: shoulders, hips, knees  -> 315, 0, 45
        "shoulder_L": (315, 0, 45), "shoulder_R": (315, 0, 45),
        "hip_L": (315, 0, 45), "hip_R": (315, 0, 45),
        "knee_L": (315, 0, 45), "knee_R": (315, 0, 45),
        # Right Extremities -> 0, 45, 90, 135, 180
        "elbow_R": (0, 45, 90, 135, 180),
        "wrist_R": (0, 45, 90, 135, 180),
        "ankle_R": (0, 45, 90, 135, 180),
        # Left Extremities -> 0, 180, 225, 270, 315
        "elbow_L": (0, 180, 225, 270, 315),
        "wrist_L": (0, 180, 225, 270, 315),
        "ankle_L": (0, 180, 225, 270, 315),
    })

    # Kinematic Inheritance Hierarchy (Paper Table II)
    kinematic_parents: Dict[str, str] = field(default_factory=lambda: {
        "wrist_L": "elbow_L", "wrist_R": "elbow_R",
        "elbow_L": "shoulder_L", "elbow_R": "shoulder_R",
        "ankle_L": "knee_L", "ankle_R": "knee_R",
        "knee_L": "hip_L", "knee_R": "hip_R",
    })

    # Cranial offset (Paper Eq. 3)
    cranial_offset_ratio: float = 0.6  # crown = 0.6 * (nose-to-shoulder) above nose

    # Volumetric Master Engine (Paper Sec. IV-F)
    y_microgrid_num_rays: int = 5
    z_microgrid_num_rays: int = 5
    y_microgrid_pad_fraction: float = 0.05  # 5% of shoulder width
    z_microgrid_pad_fraction: float = 0.02  # 2% of shoulder width

    # Ablation toggles
    use_jurisdiction_filter: bool = True
    use_volumetric_phase2: bool = True
    use_anatomical_caps: bool = True
    # ── FIX #2: Default is single-angle, NOT LS ──
    # Empirically better on real MediaPipe noise (heteroscedastic).
    # LS is optimal only under i.i.d. Gaussian noise (Theorem 4 assumption),
    # which MediaPipe violates. Set True only for the ablation study.
    use_ls_estimator: bool = False


# =====================================================================
# Phase 1.1-1.2: MediaPipe + Orthographic Camera + Semantic Filtering
# =====================================================================
class CortexVision:
    """Phase 1 of the pipeline: multi-view 2D pose extraction with
    Semantic Jurisdiction Zone filtering."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._mp_holistic = None
        self._vis = None
        self._mesh = None
        self._front_image = None
        self._init_mediapipe()

    def _init_mediapipe(self):                     # ← 4 spaces = class method
        """FIX #8: fail fast with instructions instead of a raw traceback."""
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as e:
            raise ImportError(
                "MediaPipe is required for Phase 1 (CortexVision).\n"
                "Install it with:  pip install mediapipe\n"
                "There is intentionally NO synthetic fallback: a silent "
                "fallback would produce plausible-looking garbage skeletons."
            ) from e
        self._mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=self.config.model_complexity,
            refine_face_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
        )

    def init_visualizer(self, mesh):
        """Initialize the Open3D visualizer and orthographic camera ONCE.
        Logic taken exactly from voidx_cortex_vision.py"""
        import open3d as o3d  # type: ignore
        
        self._mesh = o3d.geometry.TriangleMesh(mesh)
        self._mesh.compute_vertex_normals()
        self._mesh.paint_uniform_color(list(self.config.mesh_skin_tone))

        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(visible=False, 
                                width=self.config.render_resolution, 
                                height=self.config.render_resolution)
        self._vis.add_geometry(self._mesh)

        opt = self._vis.get_render_option()
        opt.background_color = np.asarray(self.config.background_color)

        # --- ORTHOGRAPHIC CAMERA SETUP (from old script) ---
        ctr = self._vis.get_view_control()
        bbox = self._mesh.get_axis_aligned_bounding_box()
        ctr.set_lookat(bbox.get_center())
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.8)

        params = ctr.convert_to_pinhole_camera_parameters()
        fx = params.intrinsic.intrinsic_matrix[0, 0]
        fy = params.intrinsic.intrinsic_matrix[1, 1]
        s = self.config.orthographic_scale_factor
        params.intrinsic.set_intrinsics(
            self.config.render_resolution, self.config.render_resolution,
            fx * s, fy * s,
            self.config.render_resolution / 2.0,
            self.config.render_resolution / 2.0,
        )
        ext = params.extrinsic.copy()
        ext[0, 3] *= s
        ext[1, 3] *= s
        ext[2, 3] *= s
        params.extrinsic = ext
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    def render_and_detect(self, angle_deg: int) -> Dict[str, Tuple[float, float, float]]:
        """Render mesh rotated by angle_deg, run MediaPipe, return detections."""
        import copy
        import cv2  # type: ignore

        # ── FIX #1: Deep-copy so rotation doesn't accumulate on self._mesh ──
        # normalize_normals() returns self._mesh (in-place), so without a copy,
        # each rotate() call adds to the previous rotation. By angle 315°, the
        # mesh would be at 945° instead of 315°.
        mesh_copy = copy.deepcopy(self._mesh)

        R = mesh_copy.get_rotation_matrix_from_xyz((0, math.radians(angle_deg), 0))
        mesh_copy.rotate(R, center=(0, 0, 0))

        self._vis.clear_geometries()
        self._vis.add_geometry(mesh_copy)
        self._vis.poll_events()
        self._vis.update_renderer()

        temp_img_path = f"temp_cortex_{angle_deg}.png"
        self._vis.capture_screen_image(temp_img_path)

        image = cv2.imread(temp_img_path)
        if image is None:
            print(f"Error: Failed to capture image for angle {angle_deg}")
            return {}

        # ── FIX #5 (preview): store front view in memory, not on disk ──
        if angle_deg == 0:
            self._front_image = image.copy()

        try:
            os.remove(temp_img_path)
        except Exception:
            pass

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._mp_holistic.process(image_rgb)

        detections = {}
        if results.pose_landmarks:
            for joint_name, idx in self.config.mediapipe_indices.items():
                if idx >= len(results.pose_landmarks.landmark):
                    continue
                lm = results.pose_landmarks.landmark[idx]
                x_screen = lm.x - 0.5
                y_screen = -(lm.y - 0.5)
                detections[joint_name] = (x_screen, y_screen, lm.visibility)
        return detections

    def apply_semantic_jurisdiction(self, detections: Dict[str, Tuple[float, float, float]], angle_deg: int) -> Dict[str, Tuple[float, float, float]]:
        """Filter detections by Semantic Jurisdiction Zones (Table I)."""
        if not self.config.use_jurisdiction_filter:
            return detections
        filtered = {}
        for joint, obs in detections.items():
            allowed = self.config.jurisdiction_zones.get(joint, self.config.scan_angles_deg)
            if angle_deg in allowed:
                filtered[joint] = obs
        return filtered

    def destroy(self):
        """FIX XNNPACK (1/2): tear down BOTH the visualizer and the
        MediaPipe session. Each mp.solutions.holistic.Holistic holds a
        TFLite interpreter with an XNNPACK delegate; the ablation runner
        creates ~13 pipelines per mesh, and without close() the delegates
        accumulate until the runtime fails (~27th instance). run()
        re-initializes MediaPipe when the pipeline is reused."""
        if self._vis:
            self._vis.destroy_window()
            self._vis = None
        if self._mp_holistic is not None:
            self._mp_holistic.close()
            self._mp_holistic = None

# =====================================================================
# Phase 1.3-1.5: Decoupled Axis Projection + Kinematic Inheritance + Scaling
# =====================================================================
# =====================================================================
# Phase 1.3-1.5: Decoupled Axis Projection + Kinematic Inheritance + Scaling
# =====================================================================
class DecoupledAxisProjection:
    """Recovers (X, Y, Z) for each joint from filtered multi-view
    observations using the linear least-squares estimator of Theorem 4."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def estimate_joint(                                  # ← 4 spaces
        self,
        joint_name: str,
        observations: List[Tuple[int, float, float]],
        sigma: float = 0.05,
    ) -> Optional[Dict[str, float]]:
        if len(observations) == 0:
            return None

        angles = [o[0] for o in observations]
        x_screens = [o[1] for o in observations]
        y_screens = [o[2] for o in observations]

        # Y-axis: Corollary 1 — observable from ANY number of views
        Y_hat = corollary_1_y_decouple(y_screens)

        # ── FIX #6: single-view degraded mode ──
        if len(observations) == 1:
            theta = math.radians(angles[0])
            c, s = math.cos(theta), math.sin(theta)
            if abs(c) >= abs(s):          # front/back-ish view: X observable
                X_hat = x_screens[0] / c
                Z_hat = 0.0
            else:                          # side-ish view: Z observable
                Z_hat = x_screens[0] / s
                X_hat = 0.0
            return {"x": X_hat, "y": Y_hat, "z": Z_hat,
                    "var_X": float("inf"), "var_Z": float("inf"),
                    "num_views": 1}

        # (X, Z): N >= 2  (ONE copy only)
        if self.config.use_ls_estimator:
            X_hat, Z_hat, var_X, var_Z = theorem_4_ls_optimal(
                angles, x_screens, sigma)
        else:
            X_hat, Z_hat, var_X, var_Z = self._single_angle_inversion(
                angles, x_screens)

        return {"x": X_hat, "y": Y_hat, "z": Z_hat,
                "var_X": float(var_X), "var_Z": float(var_Z),
                "num_views": len(observations)}

    def _single_angle_inversion(
        self,
        angles: List[int],
        x_screens: List[float],
    ) -> Tuple[Optional[float], Optional[float], float, float]:
        # Extract true X from frontal views only
        frontal_x = [x for a, x in zip(angles, x_screens) if a in (0, 360)]
        X_hat = float(np.mean(frontal_x)) if frontal_x else None

        # Extract true Z via trig inversion
        if X_hat is not None:
            z_estimates = []
            for a, x in zip(angles, x_screens):
                if a in (0, 180, 360):
                    continue
                rad = math.radians(a)
                z = (x - X_hat * math.cos(rad)) / math.sin(rad)
                z_estimates.append(z)
            Z_hat = float(np.mean(z_estimates)) if z_estimates else None
        else:
            Z_hat = None

        return X_hat, Z_hat, float("inf"), float("inf")

    def kinematic_inheritance(
        self,
        skeleton: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        for _ in range(3):
            for child, parent in self.config.kinematic_parents.items():
                if child not in skeleton or parent not in skeleton:
                    continue
                if skeleton[child].get("x") is None:
                    skeleton[child]["x"] = skeleton[parent].get("x", 0.0)
                if skeleton[child].get("z") is None:
                    skeleton[child]["z"] = skeleton[parent].get("z", 0.0)
        for joint in skeleton.values():
            for k in ("x", "y", "z"):
                if joint.get(k) is None:
                    joint[k] = 0.0
        return skeleton

    def cranial_offset_and_scaling(
        self,
        skeleton: Dict[str, Dict[str, float]],
        mesh_bounds: Tuple[np.ndarray, np.ndarray],
        front_image=None,
        render_resolution: int = 1024,
    ) -> Dict[str, Dict[str, float]]:
        """Apply cranial offset (Eq. 3) and dynamic Y-interpolation (Eq. 4)."""
        import cv2

        mesh_min, mesh_max = mesh_bounds
        mesh_height = float(mesh_max[1] - mesh_min[1])

        skel_pts = np.array([[j["x"], j["y"], j["z"]] for j in skeleton.values()])
        skel_min_y = float(skel_pts[:, 1].min())
        skel_max_y = float(skel_pts[:, 1].max())

        true_skel_min_y = skel_min_y
        true_skel_max_y = skel_max_y

        if front_image is not None:
            try:
                if len(front_image.shape) == 3:
                    gray = cv2.cvtColor(front_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = front_image
                ys, _ = np.where(gray > 5)
                if len(ys) > 0:
                    top_pixel_y = ys.min()
                    bottom_pixel_y = ys.max()
                    true_skel_max_y = 0.5 - (top_pixel_y / float(render_resolution))
                    true_skel_min_y = 0.5 - (bottom_pixel_y / float(render_resolution))
            except Exception as e:
                print(f"Warning: silhouette detection failed ({e}). Falling back to cranial offset.")
        else:
            print("Warning: no front image captured. Falling back to cranial offset.")

        true_skel_height = true_skel_max_y - true_skel_min_y
        if true_skel_height <= 0:
            true_skel_height = max(skel_max_y - skel_min_y, 1e-6)

        S = mesh_height / true_skel_height if true_skel_height > 0 else 1.0
        mesh_center = (mesh_min + mesh_max) / 2.0

        for name, pos in skeleton.items():
            pos["x"] = mesh_center[0] + (pos["x"] * S)
            pos["z"] = mesh_center[2] + (pos["z"] * S)

            y_from_floor = pos["y"] - true_skel_min_y
            y_ratio = y_from_floor / true_skel_height if true_skel_height > 0 else 0
            pos["y"] = float(y_ratio * mesh_height + mesh_min[1])

        if "hip_L" in skeleton and "hip_R" in skeleton:
            skeleton["pelvis"] = {
                k: 0.5 * (skeleton["hip_L"][k] + skeleton["hip_R"][k])
                for k in ("x", "y", "z")
            }
        if "shoulder_L" in skeleton and "shoulder_R" in skeleton:
            skeleton["neck"] = {
                k: 0.5 * (skeleton["shoulder_L"][k] + skeleton["shoulder_R"][k])
                for k in ("x", "y", "z")
            }
        return skeleton

# =====================================================================
# Phase 2: Volumetric Master Engine
# =====================================================================
class VolumetricMasterEngine:
    """Phase 2 of the pipeline: anchors joints to the true volumetric
    centre of the mesh via raycasting.

    Implements Sec. IV-F (Eqs. 5-6, Proposition 7-8).
    """

    def __init__(self, config: PipelineConfig, trimesh_mesh):
        self.config = config
        self.mesh = trimesh_mesh
        import trimesh  # type: ignore
        from trimesh.proximity import ProximityQuery  # type: ignore
        self._proximity = ProximityQuery(trimesh_mesh)
        self._bounds = trimesh_mesh.bounds
        self._shoulder_width = self._compute_shoulder_width()

    def _compute_shoulder_width(self) -> float:
        """Dynamic shoulder-width anchor (Sec. IV-F intro)."""
        # If a skeleton is provided later, use the true shoulder distance;
        # otherwise fall back to mesh width.
        return float(self._bounds[1][0] - self._bounds[0][0]) * 0.4

    def set_shoulder_width_from_skeleton(self, skeleton):
        sl = skeleton.get("shoulder_L")
        sr = skeleton.get("shoulder_R")
        if sl and sr:
            w = abs(sl["x"] - sr["x"])
            if w > 1e-3:
                self._shoulder_width = w

    # ---- Step 1: Proximity Engine (surface initialisation) ----
    def proximity_engine(self, x: float, y: float, z: float) -> float:
        """Find the nearest mesh surface point to ``(x, y, z)`` and return
        its Z-coordinate.

        Implements Sec. IV-F.1: the AI's Z-coordinate hovers inside the
        surface skin, so we initialise to the true physical skin depth
        ``Z_physical`` before raycasting.
        """
        try:
            closest, _, _ = self._proximity.on_surface(
                np.array([[x, y, z]])
            )
            if len(closest) > 0:
                return float(closest[0][2])
        except Exception:
            pass
        return float(z)

    # ---- Step 2: Y-axis volumetric centering (Eq. 5) ----
    def y_volumetric_centering(
        self,
        x: float,
        z_physical: float,
    ) -> float:
        """5-ray downward (-Y) micro-grid to find the volumetric Y-centre.

        Implements Eq. (5) and Proposition 7:

            Y_target = Y_bottom + |Y_top - Y_bottom| / 2

        Five parallel rays are fired at offset Z coordinates to robustly
        estimate the limb cross-section.
        """
        z_pad = self.config.y_microgrid_pad_fraction * self._shoulder_width
        laser_origin_y = float(self._bounds[1][1]) + 50.0
        origins = np.array([
            [x, laser_origin_y, z_physical + dz]
            for dz in (-2 * z_pad, -z_pad, 0, z_pad, 2 * z_pad)
        ])
        directions = np.array([[0, -1, 0]] * 5)
        try:
            locations, index_ray, _ = self.mesh.ray.intersects_location(
                ray_origins=origins, ray_directions=directions
            )
        except Exception:
            return float("nan")

        valid_centres = []
        for i in range(5):
            hits = locations[index_ray == i]
            if len(hits) < 2:
                continue
            ys = np.sort(hits[:, 1])
            y_bottom = float(ys[0])
            y_top = float(ys[-1])
            y_target, _ = proposition_7_midpoint(y_top, y_bottom)
            valid_centres.append(y_target)

        if not valid_centres:
            return float("nan")
        return float(np.mean(valid_centres))

    # ---- Step 3: Z-axis micro-grid with anatomical thickness caps (Eq. 6) ----
    def z_microgrid_with_caps(
        self,
        joint_name: str,
        x: float,
        y_target: float,
    ) -> float:
        """5-ray forward (+Z) micro-grid with anatomical thickness caps.

        Implements Eq. (6) and Proposition 8:

            Z_centered = Z_front_skin + min(dZ, dZ_max_cap) / 2

        where the cap is selected from the joint's anatomical group:
            wrists/ankles -> 15% of shoulder width
            elbows/knees  -> 25% of shoulder width
            core/torso    -> 60% of shoulder width
        """
        pad = self.config.z_microgrid_pad_fraction * self._shoulder_width
        laser_origin_z = float(self._bounds[0][2]) - 50.0
        origins = np.array([
            [x + dx, y_target + dy, laser_origin_z]
            for dx, dy in [(0, 0), (pad, 0), (-pad, 0), (0, pad), (0, -pad)]
        ])
        directions = np.array([[0, 0, 1]] * 5)
        try:
            locations, index_ray, _ = self.mesh.ray.intersects_location(
                ray_origins=origins, ray_directions=directions
            )
        except Exception:
            return float("nan")

        max_cap = (
            thickness_cap_for_joint(joint_name, self._shoulder_width)
            if self.config.use_anatomical_caps
            else float("inf")
        )

        valid_centres = []
        for i in range(5):
            hits = locations[index_ray == i]
            if len(hits) < 2:
                continue
            zs = np.sort(hits[:, 2])
            front_z = float(zs[0])
            back_z = float(zs[-1])
            thickness = abs(back_z - front_z)
            thickness = min(thickness, max_cap)
            valid_centres.append(front_z + thickness * 0.5)

        if not valid_centres:
            # --- RESTORE TIER 2: Use first available hit ---
            if len(locations) > 0:
                return float(locations[0][2])
            return float("nan")
        return float(np.mean(valid_centres))

    def parent_fallback_z(        # 4 spaces
        self,                     # 8 spaces
        joint_name: str,
        skeleton: Dict[str, Dict[str, float]],
        target_x: float,
        target_y: float,
        fallback_z: float,        # 8 spaces
    ) -> float:                   # FIXED: 4 spaces
        parent_name = self.config.kinematic_parents.get(joint_name) # FIXED: 8 spaces
        if parent_name and parent_name in skeleton:                 # FIXED: 8 spaces
            safe_z = skeleton[parent_name].get("z", fallback_z)
        else:
            safe_z = fallback_z     # ← Use Phase 1 Z, not 0.0
        try:
            closest, _, _ = self._proximity.on_surface(
                np.array([[target_x, target_y, safe_z]])
            )
            if len(closest) > 0:
                return float(closest[0][2])  # FIXED: 12 spaces
        except Exception:
            pass
        return float(safe_z)                # FIXED: 8 spaces


# =====================================================================
# Top-level pipeline (Algorithm 1)
# =====================================================================
class VoidXPipeline:
    """End-to-end deterministic volumetric auto-rigging pipeline.

    Realises Algorithm 1 of the paper.  Usage::

        config = PipelineConfig()
        pipe = VoidXPipeline(config)
        skeleton = pipe.run("path/to/mesh.obj")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.cortex = CortexVision(self.config)
        self.axis_proj = DecoupledAxisProjection(self.config)
        self._mesh = None
        self._trimesh_mesh = None

    def load_mesh(self, mesh_path: str):
        """Load via the shared loader, then build the Open3D mesh FROM the
        trimesh data (Fix #4) so both share one coordinate frame."""
        import open3d as o3d

        self._trimesh_mesh = load_mesh_trimesh(mesh_path)

        self._mesh = o3d.geometry.TriangleMesh()
        self._mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(self._trimesh_mesh.vertices, dtype=np.float64))
        self._mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(self._trimesh_mesh.faces, dtype=np.int32))
        self._mesh.compute_vertex_normals()

        self.cortex.init_visualizer(self._mesh)
        return self._trimesh_mesh

    def run(self, mesh_path: str) -> Dict:
        """Execute the full pipeline on a single mesh.

        Returns the volumetrically-centered skeleton as a dict
        ``{"pose": {joint_name: {"x":..,"y":..,"z":..}}, "metadata": {...}}``.
        """
                # FIX XNNPACK (2/2): re-init MediaPipe if a previous destroy()
        # closed it (pipeline reuse in the main experiment). Before
        # t_start: model loading is setup, not per-character inference —
        # keeps runtime metrics comparable.
        if self.cortex._mp_holistic is None:
            self.cortex._init_mediapipe()
        t_start = time.time()               # TIMESTAMP
        
        mesh = self.load_mesh(mesh_path)
        t_load = time.time()                # TIMESTAMP

        # Reset front image so a failed angle-0 capture can't reuse the
        # previous mesh's silhouette.
        self.cortex._front_image = None

        # ---- Phase 1.1-1.2: Multi-view scanning + jurisdiction filter ----
        fusion_vault: Dict[str, List[Tuple[int, float, float]]] = {
            j: [] for j in self.config.core_joints
        }
        for angle in self.config.scan_angles_deg:
            detections = self.cortex.render_and_detect(angle)
            filtered = self.cortex.apply_semantic_jurisdiction(detections, angle)
            for joint, (xs, ys, vis) in filtered.items():
                if vis > 0.1 and joint in fusion_vault:
                    fusion_vault[joint].append((angle, xs, ys))

        self.cortex.destroy()

        # ---- Phase 1.3: Decoupled Axis Projection ----
        skeleton: Dict[str, Dict[str, float]] = {}
        for joint, obs in fusion_vault.items():
            est = self.axis_proj.estimate_joint(joint, obs)
            if est is None:
                skeleton[joint] = {"x": None, "y": None, "z": None}
            else:
                skeleton[joint] = est

        # ---- Phase 1.4: Kinematic Inheritance ----
        skeleton = self.axis_proj.kinematic_inheritance(skeleton)

        # ---- Phase 1.5: Cranial offset + dynamic scaling ----
        o3d_bbox = self._mesh.get_axis_aligned_bounding_box()
        mesh_bounds = (np.asarray(o3d_bbox.get_min_bound()),
                       np.asarray(o3d_bbox.get_max_bound()))
        skeleton = self.axis_proj.cranial_offset_and_scaling(
            skeleton, mesh_bounds,
            front_image=self.cortex._front_image,
            render_resolution=self.config.render_resolution,
        )
        t_phase1 = time.time()              # TIMESTAMP (FIX: no "- t_load" here!)

        # ---- Phase 2: Volumetric Master Engine ----
        MACRO_BONES = {
            "pelvis", "neck",
            "shoulder_L", "shoulder_R",
            "elbow_L", "elbow_R",
            "wrist_L", "wrist_R",
            "hip_L", "hip_R",
            "knee_L", "knee_R",
            "ankle_L", "ankle_R",
        }

        if self.config.use_volumetric_phase2:
            engine = VolumetricMasterEngine(self.config, self._trimesh_mesh)
            engine.set_shoulder_width_from_skeleton(skeleton)
            for joint_name, pos in skeleton.items():
                if joint_name not in MACRO_BONES:
                    continue

                target_x = pos["x"]
                target_y = pos["y"]
                fallback_z = pos["z"]

                # Step 1: Proximity Engine
                z_physical = engine.proximity_engine(target_x, target_y, fallback_z)

                # Step 2: Y volumetric centering (only for horizontal limbs)
                if "elbow" in joint_name or "wrist" in joint_name:
                    y_new = engine.y_volumetric_centering(target_x, z_physical)
                    if not math.isnan(y_new):
                        target_y = y_new

                # Step 3: Z micro-grid + caps
                z_new = engine.z_microgrid_with_caps(joint_name, target_x, target_y)
                if math.isnan(z_new):
                    z_new = engine.parent_fallback_z(
                        joint_name, skeleton, target_x, target_y, fallback_z
                    )

                skeleton[joint_name] = {
                    "x": float(target_x),
                    "y": float(target_y),
                    "z": float(z_new),
                }

        joints_to_remove = [name for name in skeleton if name not in MACRO_BONES]
        for name in joints_to_remove:
            del skeleton[name]

        t_phase2 = time.time()              # TIMESTAMP
        return {
            "pose": skeleton,
            "metadata": {
                "mesh_path": mesh_path,
                "runtime_seconds": round(t_phase2 - t_start, 3),
                "load_seconds": round(t_load - t_start, 3),
                "phase1_seconds": round(t_phase1 - t_load, 3),
                "phase2_seconds": round(t_phase2 - t_phase1, 3),
                "config": {
                    "use_ls_estimator": self.config.use_ls_estimator,
                    "use_jurisdiction_filter": self.config.use_jurisdiction_filter,
                    "use_volumetric_phase2": self.config.use_volumetric_phase2,
                    "use_anatomical_caps": self.config.use_anatomical_caps,
                    "num_scan_angles": len(self.config.scan_angles_deg),
                },
                "num_joints": len(skeleton),
            },
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python voidx_pipeline.py <mesh.obj>")
        sys.exit(1)

    mesh_path = sys.argv[1]
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    print(f"Running VoidX Pipeline on: {mesh_path}...")
    pipe = VoidXPipeline()
    result = pipe.run(mesh_path)

    output_filename = "VOIDX_ULTIMATE_DNA.json"
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"\nSUCCESS! Skeleton saved to: {output_filename}")