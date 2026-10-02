"""
voidx_ablation.py
=================
Ablation study runner implementing the five ablations reported in
Section VI-E (Tables VII-XI) of the paper:

    1. Number of viewing angles N          (Table VII)
    2. Depth estimator: LS vs single-angle (Table VIII)
    3. Semantic Jurisdiction Zones on/off  (Section VI-E, paragraph 3)
    4. Volumetric Master Engine on/off     (Section VI-E, paragraph 4)
    5. Volumetric centering method         (Section VI-E, paragraph 5)

Each ablation function takes a base pipeline config and a mesh, varies
exactly one design choice, runs the pipeline, and returns a list of
result dicts ready for tabulation.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List

import numpy as np

from voidx_pipeline import PipelineConfig, VoidXPipeline
from voidx_metrics import compute_all_metrics


# =====================================================================
# Ablation 1: Number of viewing angles N (Table VII)
# =====================================================================
def ablation_num_views(
    mesh_path: str,
    gt_skeleton: Dict,
    mesh,
    N_values: List[int] = (1, 2, 4, 8, 16),
) -> List[Dict]:
    """Compare the default N=8 viewing angles (Section V-A) against
    a range of other N values. Returns a list of result dicts.  
    """
    results = []
    for N in N_values:
        if N == 1:
            angles = (90,)
        else:
            angles = tuple(int(a) for a in np.linspace(0, 360, N, endpoint=False))
        cfg = PipelineConfig(scan_angles_deg=angles)
        pipe = VoidXPipeline(cfg)
        t0 = time.time()
        pred = pipe.run(mesh_path)
        elapsed = time.time() - t0
        metrics = compute_all_metrics(pred["pose"], gt_skeleton, mesh)
        metrics["N"] = N
        metrics["Time_s"] = round(elapsed, 2)
        results.append(metrics)
    return results


# =====================================================================
# Ablation 2: Depth estimator choice (Table VIII)
# =====================================================================
def ablation_estimator(
    mesh_path: str,
    gt_skeleton: Dict,
    mesh,
) -> List[Dict]:
    
    results = []
    configs = [
        ("Decoupled Trig Inversion (Ours)", {"use_ls_estimator": False}),
        ("Linear least squares (LS)", {"use_ls_estimator": True}),
    ]
    for name, overrides in configs:
        cfg = PipelineConfig(**overrides)
        pipe = VoidXPipeline(cfg)
        pred = pipe.run(mesh_path)
        metrics = compute_all_metrics(pred["pose"], gt_skeleton, mesh)
        metrics["Estimator"] = name
        results.append(metrics)
    return results


# =====================================================================
# Ablation 3: Semantic Jurisdiction Zones on/off
# =====================================================================
def ablation_jurisdiction(
    mesh_path: str,
    gt_skeleton: Dict,
    mesh,
) -> List[Dict]:
    """Compare the default jurisdiction zones (Section V-B) against
    a baseline that ignores jurisdiction entirely (i.e. all joints are
    allowed to pierce any geometry).
    """
    results = []
    for label, enabled in [("Jurisdiction ON", True), ("Jurisdiction OFF", False)]:
        cfg = PipelineConfig(use_jurisdiction_filter=enabled)
        pipe = VoidXPipeline(cfg)
        pred = pipe.run(mesh_path)
        metrics = compute_all_metrics(pred["pose"], gt_skeleton, mesh)
        metrics["Config"] = label
        results.append(metrics)
    return results


# =====================================================================
# Ablation 4: Volumetric Master Engine on/off + caps on/off
# =====================================================================
def ablation_volumetric(
    mesh_path: str,
    gt_skeleton: Dict,
    mesh,
) -> List[Dict]:
    """Compare the default volumetric master engine (Section V-C) against
    a baseline that ignores volumetric reasoning entirely (i.e. only the surface is used to estimate depth)."""
    results = []
    configs = [
        ("Phase 2 OFF (surface only)", {
            "use_volumetric_phase2": False,
        }),
        ("Phase 2 ON, caps OFF", {
            "use_volumetric_phase2": True,
            "use_anatomical_caps": False,
        }),
        ("Phase 2 ON, caps ON (full)", {
            "use_volumetric_phase2": True,
            "use_anatomical_caps": True,
        }),
    ]
    for label, overrides in configs:
        cfg = PipelineConfig(**overrides)
        pipe = VoidXPipeline(cfg)
        pred = pipe.run(mesh_path)
        metrics = compute_all_metrics(pred["pose"], gt_skeleton, mesh)
        metrics["Config"] = label
        results.append(metrics)
    return results


# =====================================================================
# Ablation 5: Volumetric centering method (midpoint vs area centroid)
# =====================================================================
def ablation_centering_method(
    mesh_path: str,
    gt_skeleton: Dict,
    mesh,
) -> List[Dict]:
    """Compare the midpoint estimator (Eq. 5, Proposition 7) against
    the area centroid (trapezoidal integration over the 5-ray micro-grid).
    """
    results = []
    # Midpoint (default)
    cfg = PipelineConfig()
    pipe = VoidXPipeline(cfg)
    pred_midpoint = pipe.run(mesh_path)
    metrics_mid = compute_all_metrics(pred_midpoint["pose"], gt_skeleton, mesh)
    metrics_mid["Method"] = "Midpoint (Eq. 5)"
    results.append(metrics_mid)

    # Area centroid: Note - Full implementation of area centroid requires 
    # patching VolumetricMasterEngine. We skip the fake approximation and 
    # only report the midpoint for now, keeping the data 100% honest.
    # If you implement the area centroid in the engine later, you can 
    # run it here. For now, we just return the midpoint result.
    
    return results


# =====================================================================
# Driver: run all ablations
# =====================================================================
def run_all_ablations(
    mesh_path: str,
    gt_skeleton: Dict,
    mesh,
) -> Dict[str, List[Dict]]:
    """Run all five ablation studies on a single mesh.

    Returns a dict mapping ablation name to its result list.
    """
    return {
        "ablation_1_num_views": ablation_num_views(mesh_path, gt_skeleton, mesh),
        "ablation_2_estimator": ablation_estimator(mesh_path, gt_skeleton, mesh),
        "ablation_3_jurisdiction": ablation_jurisdiction(mesh_path, gt_skeleton, mesh),
        "ablation_4_volumetric": ablation_volumetric(mesh_path, gt_skeleton, mesh),
        "ablation_5_centering_method": ablation_centering_method(
            mesh_path, gt_skeleton, mesh
        ),
    }
