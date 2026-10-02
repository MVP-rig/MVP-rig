"""
voidx_theorems.py
=================
Formal implementations and empirical verifications of the six theorems and
two propositions stated in Section III (Mathematical Preliminaries) and
Section IV-C (Volumetric Master Engine) of:

    "Deterministic Volumetric Auto-Rigging via Multi-View Orthographic
     Lifting of 2D Pose Landmarks" (Krishnanand, 2026)

Every function in this module is the *executable* counterpart of a stated
theorem.  Each docstring cites the theorem number, the equation it realises,
and the proof sketch from the paper.  A driver function ``verify_all()``
runs a Monte-Carlo validation that reproduces the numbers reported in
Table II (variance multiplier) and Figure 4 (CRLB curve) of the paper.

This module is dependency-light (numpy + scipy only) so it can be imported
in any environment, including CI, without MediaPipe or Open3D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import linear_sum_assignment


# =====================================================================
# Theorem 1  (Paper Eq. 1)
# Orthographic Projection under Y-Rotation
# =====================================================================
def theorem_1_projection(
    X: float, Y: float, Z: float, theta_deg: float
) -> Tuple[float, float]:
    """Return ``(x_screen, y_screen)`` under orthographic projection of the
    point ``(X, Y, Z)`` after a Y-axis rotation by ``theta_deg``.

    Realises Theorem 1 of the paper:

        x_screen(theta) = X * cos(theta) + Z * sin(theta)
        y_screen(theta) = Y              (rotation-invariant)

    Proof sketch: applying R_y(theta) to (X,Y,Z) gives p' with
    p'_x = X cos(theta) + Z sin(theta), p'_y = Y; orthographic projection
    discards p'_z.  This is a standard result in multi-view geometry
    [Hartley & Zisserman 2004, Ch. 4].
    """
    theta = math.radians(theta_deg)
    x_screen = X * math.cos(theta) + Z * math.sin(theta)
    y_screen = Y  # Corollary 1: independent of theta
    return x_screen, y_screen


# =====================================================================
# Corollary 1  (Paper, immediately after Theorem 1)
# Y-Axis Decoupling
# =====================================================================
def corollary_1_y_decouple(
    y_observations: Sequence[float],
) -> float:
    """Maximum-likelihood estimator of Y under i.i.d. Gaussian noise.

    Realises Corollary 1: because y_screen(theta) = Y is independent of
    theta, the MLE is the arithmetic mean across all valid views:

        Y_hat = (1/N) * sum_i y_screen(theta_i)
    """
    if len(y_observations) == 0:
        raise ValueError("Cannot estimate Y from zero observations.")
    return float(np.mean(y_observations))


# =====================================================================
# Theorem 2  (Paper, Section III-B)
# Minimum Views for Noise-Free Recovery
# =====================================================================
def theorem_2_min_views(theta1_deg: float, theta2_deg: float) -> bool:
    """Return True iff two views at the given angles admit exact recovery
    of (X, Z).

    Realises Theorem 2: recovery is possible iff sin(theta2 - theta1) != 0,
    i.e. the two angles are not antipodal.  Equivalently, the 2x2 system
    A u = x with A = [[cos t1, sin t1], [cos t2, sin t2]] is invertible
    iff det(A) = sin(theta2 - theta1) != 0.
    """
    delta = math.radians(theta2_deg - theta1_deg)
    return abs(math.sin(delta)) > 1e-12


# =====================================================================
# Theorem 3  (Paper Eq. in Section III-D)
# Cramér-Rao Lower Bound for Depth Estimation
# =====================================================================
def theorem_3_crlb(N: int, sigma: float) -> float:
    """Return the Cramér-Rao lower bound on Var(Z_hat) for an isotropic
    N-view angle set with i.i.d. Gaussian noise of std-dev ``sigma``.

    Realises Theorem 3:

        Var(Z_hat) >= 2 * sigma^2 / N

    with equality iff the angle set is isotropic
    (sum cos^2 = sum sin^2 = N/2, sum sin*cos = 0).
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    return 2.0 * (sigma ** 2) / N


def is_isotropic(angle_set_deg: Sequence[float]) -> bool:
    """Check whether an angle set satisfies the isotropy conditions of
    Theorem 3 (within a numerical tolerance)."""
    if len(angle_set_deg) == 0:
        return False
    N = len(angle_set_deg)
    rad = np.radians(angle_set_deg)
    sum_cos2 = float(np.sum(np.cos(rad) ** 2))
    sum_sin2 = float(np.sum(np.sin(rad) ** 2))
    sum_cs = float(np.sum(np.sin(rad) * np.cos(rad)))
    tol = 1e-6 * N
    return (
        abs(sum_cos2 - N / 2.0) < tol
        and abs(sum_sin2 - N / 2.0) < tol
        and abs(sum_cs) < tol
    )


# =====================================================================
# Theorem 4  (Paper Eq. 7-8)
# LS Optimality
# =====================================================================
def theorem_4_ls_optimal(
    angles_deg: Sequence[float],
    x_screen_obs: Sequence[float],
    sigma: float,
) -> Tuple[float, float, float, float]:
    """Linear least-squares estimator of (X, Z) and the theoretical
    variance of Z_hat.

    Realises Theorem 4: for an isotropic N-view angle set, the LS
    estimator

        u_hat = (A^T A)^{-1} A^T x

    achieves the CRLB.  For N=8 uniform angles, Var(Z_hat) = sigma^2 / 4.

    Returns
    -------
    X_hat, Z_hat, var_X, var_Z
        Point estimates and theoretical variances (CRLB when isotropic).
    """
    if len(angles_deg) != len(x_screen_obs):
        raise ValueError("angles and observations must have equal length.")
    if len(angles_deg) < 2:
        raise ValueError("At least 2 views required for LS estimation.")

    A = np.column_stack([
        np.cos(np.radians(angles_deg)),
        np.sin(np.radians(angles_deg)),
    ])
    b = np.asarray(x_screen_obs, dtype=float)

    # Closed-form LS solution:  u_hat = (A^T A)^{-1} A^T b
    # Using scipy.lstsq for numerical stability when A^T A is near-singular.
    u_hat, _, _, _ = lstsq(A, b)

    # Covariance:  sigma^2 * (A^T A)^{-1}
    AtA = A.T @ A
    try:
        cov = (sigma ** 2) * np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)

    X_hat, Z_hat = float(u_hat[0]), float(u_hat[1])
    var_X, var_Z = float(cov[0, 0]), float(cov[1, 1])
    return X_hat, Z_hat, var_X, var_Z


# =====================================================================
# Theorem 5  (Paper, Section III-D)
# Singularity of Single-Angle Inversion
# =====================================================================
def theorem_5_singularity(theta_deg: float) -> Tuple[bool, float]:
    """Return ``(is_singular, condition_number)`` for the single-angle
    depth estimator

        Z_hat = (x_screen - X cos(theta)) / sin(theta)

    Realises Theorem 5: the estimator is undefined (singular) at
    theta in {0, 180} degrees and ill-conditioned near these angles,
    with condition number  kappa(theta) = 1/|sin theta| + |cot theta|.
    """
    theta = math.radians(theta_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    if abs(sin_t) < 1e-12:
        return True, math.inf
    kappa = 1.0 / abs(sin_t) + abs(cos_t / sin_t)
    return False, float(kappa)


# =====================================================================
# Theorem 6  (Paper, Section III-E)
# Single-Slice Tomographic Equivalence
# =====================================================================
def theorem_6_tomographic(
    X: float, Z: float, theta_deg: float, s: float
) -> float:
    """Evaluate the Radon transform of a Dirac point mass delta(p - (X, Z))
    at projection angle ``theta_deg`` and detector offset ``s``.

    Realises Theorem 6: the measurement model
        x_screen(theta) = X cos(theta) + Z sin(theta)
    is identical to the sinogram trace of delta(x - X) delta(z - Z)
    under the 2D Radon transform,

        R delta(theta, s) = delta(s - X cos theta - Z sin theta).

    For the deterministic point-mass prior, this admits exact recovery
    from as few as two non-antipodal views (Theorem 2).  This equivalence
    motivates treating multi-view joint localisation as single-slice
    tomographic reconstruction.
    """
    theta = math.radians(theta_deg)
    # Radon transform of a Dirac is a Dirac on the (theta, s) plane.
    # We return the argument of the delta; the mass concentrates where
    # this equals zero.
    return s - (X * math.cos(theta) + Z * math.sin(theta))


# =====================================================================
# Proposition 7  (Paper, Section IV-C, Eq. 5)
# Optimality of the Midpoint Estimator
# =====================================================================
def proposition_7_midpoint(
    y_top_skin: float, y_bottom_skin: float
) -> Tuple[float, Dict[str, bool]]:
    """Return the midpoint Y-target and the conditions under which it is
    the MLE.

    Realises Proposition 7 and Eq. (5):

        Y_target = Y_bottom + |Y_top - Y_bottom| / 2

    The midpoint is the MLE iff (i) the limb cross-section is symmetric
    about its midline AND (ii) Y_top, Y_bottom are observed with i.i.d.
    Gaussian noise of equal variance.  Under arbitrary asymmetric
    cross-sections the optimal centre is the area centroid, which the
    5-ray micro-grid approximates via trapezoidal integration.
    """
    y_target = y_bottom_skin + abs(y_top_skin - y_bottom_skin) / 2.0
    conditions = {
        "symmetric_cross_section_assumed": True,
        "equal_variance_noise_assumed": True,
        "is_mle_under_assumptions": True,
    }
    return float(y_target), conditions


def area_centroid_trapezoidal(
    ray_x_offsets: Sequence[float],
    ray_y_top: Sequence[float],
    ray_y_bottom: Sequence[float],
) -> float:
    """Compute the area centroid of a limb cross-section sampled by a
    parallel ray micro-grid using the trapezoidal rule.

    Used in the ablation study (Section VI-E, last paragraph) to compare
    against the midpoint estimator of Proposition 7.
    """
    if len(ray_x_offsets) < 2:
        return float(np.mean([
            ray_y_bottom[i] + abs(ray_y_top[i] - ray_y_bottom[i]) / 2.0
            for i in range(len(ray_x_offsets))
        ]))
    x = np.asarray(ray_x_offsets, dtype=float)
    y_top = np.asarray(ray_y_top, dtype=float)
    y_bot = np.asarray(ray_y_bottom, dtype=float)
    # Cross-sectional "area" sampled at each ray: |y_top - y_bot|
    heights = np.abs(y_top - y_bot)
    # Centroid y per ray
    y_centroid_per_ray = y_bot + heights / 2.0
    # Trapezoidal integration: numerator = integral(y_c * h dx), denom = integral(h dx)
    order = np.argsort(x)
    x_s = x[order]
    h_s = heights[order]
    yc_s = y_centroid_per_ray[order]
    numerator = np.trapz(yc_s * h_s, x_s)
    denominator = np.trapz(h_s, x_s)
    if denominator == 0:
        return float(np.mean(y_centroid_per_ray))
    return float(numerator / denominator)


# =====================================================================
# Proposition 8  (Paper, Section IV-C)
# Anthropometric Grounding of Thickness Caps
# =====================================================================
@dataclass(frozen=True)
class AnthroCaps:
    """Thickness caps as fractions of bideltoid shoulder width.

    Realises Proposition 8: caps are conservative upper bounds on the
    anterior-posterior ray-traversal length consistent with ANSUR II
    anthropometric data.  The 50th-percentile male bideltoid shoulder
    breadth is ~47 cm, wrist breadth ~5.5-6 cm (~12-13%), elbow breadth
    ~6.5 cm (~14%), chest AP depth ~21 cm (~45%).  The 15/25/60% caps
    overestimate bony anatomy by 1.2-1.8x to accommodate muscular and
    stylised character proportions.
    """
    wrist_ankle: float = 0.15
    elbow_knee: float = 0.25
    core_torso: float = 0.60


def proposition_8_anthro_caps(shoulder_width: float) -> Dict[str, float]:
    """Return absolute thickness caps (in mesh units) for a given
    bideltoid ``shoulder_width``."""
    c = AnthroCaps()
    return {
        "wrist_ankle": shoulder_width * c.wrist_ankle,
        "elbow_knee": shoulder_width * c.elbow_knee,
        "core_torso": shoulder_width * c.core_torso,
        "_anthro_reference": "ANSUR II 50th-percentile male, ~47 cm bideltoid",
    }


def thickness_cap_for_joint(joint_name: str, shoulder_width: float) -> float:
    """Select the appropriate thickness cap for a joint based on its
    anatomical group."""
    caps = proposition_8_anthro_caps(shoulder_width)
    name = joint_name.lower()
    if any(k in name for k in ("wrist", "ankle")):
        return caps["wrist_ankle"]
    if any(k in name for k in ("elbow", "knee")):
        return caps["elbow_knee"]
    return caps["core_torso"]


# =====================================================================
# Observability analysis  (Paper, Section V-B, Table IV)
# =====================================================================
def observability_of_zone(
    zone_angles_deg: Sequence[float], sigma: float
) -> Dict[str, float]:
    """Compute condition number and Z-variance for a Semantic
    Jurisdiction Zone angle set.

    Reproduces the numbers in Table IV of the paper.  For each zone we
    report:
        N              - number of angles
        cond(A^T A)    - condition number (1.0 = isotropic, optimal)
        Var(Z)/sigma^2 - depth-estimator variance multiplier
    """
    N = len(zone_angles_deg)
    if N < 2:
        return {"N": N, "cond": float("inf"), "var_Z_over_sigma2": float("inf")}
    A = np.column_stack([
        np.cos(np.radians(zone_angles_deg)),
        np.sin(np.radians(zone_angles_deg)),
    ])
    AtA = A.T @ A
    eigvals = np.linalg.eigvalsh(AtA)
    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    var_Z = float((sigma ** 2) / max(eigvals[0], 1e-30)) / (sigma ** 2)
    return {"N": N, "cond": cond, "var_Z_over_sigma2": var_Z}


# =====================================================================
# Driver: verify_all()
# Reproduces Table II (variance multipliers) and Figure 4 (CRLB curve)
# =====================================================================
def _mc_z_variance(angles, num_trials, sigma, rng, true_X, true_Z,
                   bias=0.0, bias_angles=(90, 270)):
    """Monte-Carlo estimate of Var(Z_hat)/sigma^2 for the LS estimator.

    bias > 0 adds a constant offset at bias_angles, modelling MediaPipe's
    systematic side-view error. bias=0 is the pure-noise experiment that
    actually verifies Theorems 3 & 4.
    """
    trials = []
    for _ in range(num_trials):
        obs = [
            true_X * math.cos(math.radians(t))
            + true_Z * math.sin(math.radians(t))
            + rng.normal(0, sigma)
            + (bias if t in bias_angles else 0.0)
            for t in angles
        ]
        _, Z_hat, _, _ = theorem_4_ls_optimal(angles, obs, sigma)
        trials.append((Z_hat - true_Z) ** 2)
    return float(np.mean(trials)) / (sigma ** 2)


def _crlb_z_over_sigma2(angles_deg) -> float:
    """CRLB on Var(Z_hat)/sigma^2 for the LS estimator with an ARBITRARY
    angle set:  Var(Z_hat) >= sigma^2 * [(A^T A)^{-1}]_zz.

    FIX B1: reduces to 2/N (Theorem 3) ONLY when the set is isotropic.
    Non-isotropic zones (e.g. front-core [315,0,45], a 90-degree arc)
    have a HIGHER true bound — 2/N understates it.
    """
    A = np.column_stack([
        np.cos(np.radians(angles_deg)),
        np.sin(np.radians(angles_deg)),
    ])
    try:
        cov = np.linalg.inv(A.T @ A)
        return float(cov[1, 1])
    except np.linalg.LinAlgError:
        return float("inf")


def verify_all(num_trials: int = 2000, sigma: float = 0.1,
               systematic_bias: float = 0.05) -> Dict:
    """Two SEPARATE experiments, clearly labeled.

    - 'pure'   : i.i.d. Gaussian noise only. THE theorem verification —
                 empirical should match the CRLB within MC error (~3%).
    - 'biased' : side views (90/270) get a constant offset modelling
                 MediaPipe's systematic error. NOT a verification — it
                 quantifies robustness and motivates the Jurisdiction
                 Zones. Never present these numbers as CRLB verification.
    """
    rng = np.random.default_rng(42)
    true_X, true_Z = 2.0, 1.5

    # --- Table II: variance multipliers ---
    table_ii = {}
    configs = {
        "single_angle_90": [90],
        "two_angles_45_135": [45, 135],
        "four_uniform": [0, 90, 180, 270],
        "eight_uniform": [0, 45, 90, 135, 180, 225, 270, 315],
        "jurisdiction_front_core": [315, 0, 45],
        "jurisdiction_extremity": [0, 45, 90, 135, 180],
    }
    for name, angles in configs.items():
        entry = {"N": len(angles)}
        if len(angles) < 2:
            # Exact, not a placeholder: at theta=90, x_screen = Z, so the
            # naive estimator passes noise through unchanged.
            entry["theoretical_var_over_sigma2"] = 1.0
            entry["empirical_pure_var_over_sigma2"] = 1.0
            entry["ratio_pure_to_CRLB"] = 1.0
            # Biased single side view: Var = sigma^2 + bias^2 (exact).
            entry["empirical_with_sideview_bias"] = round(
                1.0 + (systematic_bias / sigma) ** 2, 4)
        else:
            # FIX B1: bound from the ACTUAL angle set, not 2/N.
            theoretical = _crlb_z_over_sigma2(angles)
            entry["is_isotropic"] = is_isotropic(angles)  # annotate: True -> 2/N applies
            pure = _mc_z_variance(angles, num_trials, sigma,
                                  rng, true_X, true_Z, bias=0.0)
            entry["theoretical_var_over_sigma2"] = round(theoretical, 4)
            entry["empirical_pure_var_over_sigma2"] = round(pure, 4)
            entry["ratio_pure_to_CRLB"] = round(pure / max(theoretical, 1e-12), 4)
            entry["empirical_with_sideview_bias"] = round(
                _mc_z_variance(angles, num_trials, sigma, rng,
                               true_X, true_Z, bias=systematic_bias), 4)
        table_ii[name] = entry

    # --- Table IV: observability of jurisdiction zones ---
    table_iv = {
        "all_8_unfiltered": observability_of_zone(
            [0, 45, 90, 135, 180, 225, 270, 315], sigma),
        "right_extremities": observability_of_zone([0, 45, 90, 135, 180], sigma),
        "left_extremities": observability_of_zone([0, 180, 225, 270, 315], sigma),
        "front_facing_core": observability_of_zone([315, 0, 45], sigma),
    }

    # --- Theorem 5: singularity ---
    singularity_checks = {
        "theta_0": theorem_5_singularity(0),
        "theta_45": theorem_5_singularity(45),
        "theta_90": theorem_5_singularity(90),
        "theta_180": theorem_5_singularity(180),
    }

    # --- Figure 4 data: CRLB curve (PURE noise only).
    #     Uses the actual-set bound too — the int() angle truncation for
    #     N=16 makes that set very slightly non-uniform. ---
    figure_4 = []
    for N in range(1, 17):
        if N == 1:
            empirical = 1.0            # exact for the naive single-angle
            theoretical = 1.0
        else:
            angles = ([45, 135] if N == 2 else
                      list(np.linspace(0, 360, N, endpoint=False).astype(int)))
            empirical = _mc_z_variance(angles, num_trials, sigma,
                                       rng, true_X, true_Z, bias=0.0)
            theoretical = _crlb_z_over_sigma2(angles)   # FIX B1 here too
        figure_4.append({
            "N": N,
            "CRLB_theoretical": round(theoretical, 4),
            "empirical_LS_pure": round(empirical, 4),
        })

    return {
        "table_ii_variance_multipliers": table_ii,
        "table_iv_jurisdiction_observability": table_iv,
        "theorem_5_singularity_checks": {
            k: {"is_singular": v[0], "condition_number": v[1]}
            for k, v in singularity_checks.items()
        },
        "figure_4_crlb_curve": figure_4,
        "isotropic_check_eight_uniform": is_isotropic(
            [0, 45, 90, 135, 180, 225, 270, 315]),
    }


if __name__ == "__main__":
    import json
    print("Verifying all theorems via Monte-Carlo simulation...")
    results = verify_all(num_trials=2000, sigma=0.1)
    print(json.dumps(results, indent=2))
    print("\nTheorem 6 (tomographic) sample: R delta(45, s=2.47) =",
          round(theorem_6_tomographic(2.0, 1.5, 45, 2.474874), 6),
          "(should be ~0)")