"""
voidx_experiment.py
===================
Main experiment runner that orchestrates the full evaluation reported
in Section VI of the paper:

    1. Load the RigNetv1 test split (humanoid subset,72 characters)
       or fall back to the user-supplied local mesh set.
    2. Run the VoidXPipeline on each mesh.
    3. Compute the RigNet metric suite (CD-J2J, CD-J2B, CD-B2B, IoU,
       Precision, Recall, ED, VCE).
    4. Run the 5 ablation studies on a representative subset.
    5. Measure end-to-end runtime per character.
    6. Output a results table (CSV + JSON) reproducing Tables V-XI of
       the paper.

Usage:
    python voidx_experiment.py --rignet_root /path/to/rignet/test \
                               --output_dir ./results
    python voidx_experiment.py --mesh_dir /path/to/meshes \
                               --output_dir ./results
"""

from __future__ import annotations

import math
import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from voidx_pipeline import PipelineConfig, VoidXPipeline, load_mesh_trimesh
from voidx_metrics import compute_all_metrics, volumetric_centering_error
from voidx_ablation import run_all_ablations
from voidx_theorems import verify_all
def _finite(x):
    """True if x is a real, finite number (False for None/nan/inf/str)."""
    return isinstance(x, (int, float)) and math.isfinite(x)
# =====================================================================
# Dataset loaders
# =====================================================================
def load_rignet_test_split(
    rignet_root: str,
    humanoid_only: bool = True,
) -> List[Dict]:
    """Load the RigNetv1 test split from .txt rig_info files."""
    test_split_file = os.path.join(rignet_root, "test_final.txt")
    if not os.path.exists(test_split_file):
        raise FileNotFoundError(
            f"RigNet test split not found at {test_split_file}. "
        )
    with open(test_split_file) as f:
        names = [line.strip() for line in f if line.strip()]

    items = []
    for name in names:
        obj_path = os.path.join(rignet_root, "objs", f"{name}.obj")
        # Look for the .txt file in the rig_info folder
        skel_path = os.path.join(rignet_root, "rig_info", f"{name}.txt")
        if not (os.path.exists(obj_path) and os.path.exists(skel_path)):
            continue
        items.append({"name": name, "mesh_path": obj_path, "gt_path": skel_path})

    if humanoid_only:
        # Since we can't easily check TEXT, we count the lines starting 
        # with "joints" in the txt file. Humanoids usually have 15-40 joints.
        filtered = []
        for item in items:
            try:
                with open(item["gt_path"], 'r') as f:
                    joint_count = sum(1 for line in f if line.strip().startswith("joints"))
                if 14 <= joint_count <= 50:
                    filtered.append(item)
            except Exception:
                continue
        items = filtered
    return items


# Mixamo to Internal Name Mapping
MIXAMO_MAP = {
    "Hips": "pelvis", "Spine": "spine", "Spine1": "spine1", "Spine2": "spine2",
    "Neck": "neck",
    "LeftShoulder": "shoulder_L", "RightShoulder": "shoulder_R",
    "LeftArm": "arm_L", "RightArm": "arm_R",
    "LeftForeArm": "elbow_L", "RightForeArm": "elbow_R",
    "LeftHand": "wrist_L", "RightHand": "wrist_R",
    "LeftUpLeg": "hip_L", "RightUpLeg": "hip_R",
    "LeftLeg": "knee_L", "RightLeg": "knee_R",
    "LeftFoot": "ankle_L", "RightFoot": "ankle_R",
    "LeftToeBase": "foot_L", "RightToeBase": "foot_R"
}

def _normalize_gt_json(data) -> Dict[str, List[float]]:
    """Normalize any common bone_3d.json schema to {name: [x,y,z]}.
    FIX A1: the old parser only understood one schema and returned {}
    silently on any other layout."""
    # Unwrap common container keys
    if isinstance(data, dict):
        for key in ("joints", "bones", "skeleton", "pose"):
            if key in data and isinstance(data[key], (dict, list)):
                data = data[key]
                break
    # Schema: list of records [{"name": ..., "position": [...]}, ...]
    if isinstance(data, list):
        out = {}
        for rec in data:
            if not isinstance(rec, dict):
                continue
            name = rec.get("name") or rec.get("joint") or rec.get("bone")
            pos = (rec.get("position") or rec.get("pos")
                   or rec.get("coordinates") or rec.get("location"))
            if name and isinstance(pos, (list, tuple)) and len(pos) == 3:
                out[name] = list(pos)
        return out
    # Schema: flat dict {name: [x,y,z]} or {name: {"x":..,"y":..,"z":..}}
    if isinstance(data, dict):
        out = {}
        for name, pos in data.items():
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                out[name] = list(pos)
            elif isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
                out[name] = [pos["x"], pos["y"], pos["z"]]
        return out
    return {}


def _map_joint_name(joint_name: str) -> str:
    """FIX A4: robust name mapping — handles 'mixamorig:Hips' prefixes
    and case differences, falls back to passthrough."""
    clean = joint_name.split(":")[-1].strip()      # strips 'mixamorig:'
    for mixamo, internal in MIXAMO_MAP.items():
        if mixamo.lower() == clean.lower():
            return internal
    return clean  # may already be an internal name like 'pelvis'


def load_gt_skeleton(gt_path: str) -> Dict:
    skeleton = {}

    if gt_path.endswith(".txt"):  # RigNet format (kept for compat)
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "joints":
                    internal = _map_joint_name(parts[1])
                    skeleton[internal] = {
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "z": float(parts[4]),
                    }
        return skeleton

    # JSON formats (bone_3d.json etc.)
    with open(gt_path, 'r') as f:
        raw = json.load(f)
    for name, coords in _normalize_gt_json(raw).items():
        internal = _map_joint_name(name)
        try:
            skeleton[internal] = {
                "x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])
            }
        except (ValueError, TypeError):
            continue

    # FIX A1: LOUD failure instead of silent empty dict
    if len(skeleton) == 0:
        preview = (list(raw.keys())[:5] if isinstance(raw, dict)
                   else f"list of {len(raw)} items")
        print(f"  WARNING: GT skeleton EMPTY for {gt_path}! "
              f"Unrecognized schema. Top-level: {preview}")
    return skeleton

def load_local_meshes(mesh_dir: str) -> List[Dict]:
    """Recursively load meshes + bone_3d.json GT.

    FIXES:
    - A3 (original): unique names from relative folder paths.
    - NEW FIX: when --mesh_dir points DIRECTLY at a single character
      folder (e.g., ...\\char93 containing mesh.obj), the mesh's parent
      IS the root, so relative_to(root) returned "." and the prediction
      was saved as "..json" — overwriting itself on every run. Now the
      folder name (char93) is used instead.
    - Prints the discovered character list so you can verify what will
      be processed before the pipeline starts.
    """
    priority = {".glb": 0, ".obj": 1, ".fbx": 2, ".ply": 3}
    root = Path(mesh_dir).resolve()

    # First pass: collect all mesh files
    all_meshes = [p for p in sorted(root.rglob("*"))
                  if p.suffix.lower() in priority]

    # Is this a multi-character dataset (meshes in subfolders)
    # or a single character folder (mesh directly in root)?
    has_subfolder_meshes = any(p.parent != root for p in all_meshes)

    by_name: Dict[str, Dict] = {}
    for p in all_meshes:
        if p.parent == root:
            if has_subfolder_meshes:
                # Dataset root with a stray mesh — name by file stem
                name = p.stem
            else:
                # SINGLE-CHARACTER folder (e.g., ...\\char93\\mesh.obj)
                # — use the folder name (char93), NOT "."
                name = root.name
        else:
            name = str(p.parent.relative_to(root))

        json_path = p.parent / "bone_3d.json"
        candidate = {
            "name": name,
            "mesh_path": str(p),
            "gt_path": str(json_path) if json_path.exists() else None,
        }

        existing = by_name.get(name)
        if existing is None:
            by_name[name] = candidate
        elif priority[p.suffix.lower()] < priority[
                Path(existing["mesh_path"]).suffix.lower()]:
            by_name[name] = candidate

    items = list(by_name.values())
    n_gt = sum(1 for it in items if it["gt_path"])
    print(f"Dataset: {len(items)} characters | {n_gt} with GT | "
          f"{len(items) - n_gt} WITHOUT GT")
    if n_gt == 0:
        print("WARNING: NO ground truth found — all metrics will be VCE-only!")

    # NEW: print what was discovered so you can verify before the long run
    print("Discovered characters:")
    for it in items:
        print(f"  - {it['name']:20s} {Path(it['mesh_path']).name:15s} "
              f"{'(GT found)' if it['gt_path'] else '(NO GT)'}")
    return items

# =====================================================================
# Main experiment
# =====================================================================
def run_main_experiment(dataset: List[Dict], output_dir: str)  -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    pipe = VoidXPipeline()
    all_results = []
    os.makedirs(os.path.join(output_dir, "preds"), exist_ok=True)   # NEW

    for i, item in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] Processing {item['name']}...")
        try:
            mesh = load_mesh_trimesh(item["mesh_path"])     # FIX 12
        except Exception as e:
            print(f"  Failed to load mesh: {e}")
            continue

        t0 = time.time()
        try:
            pred = pipe.run(item["mesh_path"])
        except Exception as e:
            print(f"  Pipeline failed: {e}")
            continue
        runtime = time.time() - t0
                # NEW: save prediction — metric changes never require re-running
        with open(os.path.join(output_dir, "preds", f"{item['name']}.json"), "w") as f:
            json.dump(pred, f)

        gt_skeleton = {}                                     # FIX A6
        if item["gt_path"]:
            gt_skeleton = load_gt_skeleton(item["gt_path"])
            core_joints = {
                "pelvis", "neck",
                "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
                "wrist_L", "wrist_R", "hip_L", "hip_R",
                "knee_L", "knee_R", "ankle_L", "ankle_R",
            }
            gt_skeleton = {k: v for k, v in gt_skeleton.items() if k in core_joints}

            if gt_skeleton:
                gt_arr = np.array([[v["x"], v["y"], v["z"]]
                                   for v in gt_skeleton.values()])
                bmin, bmax = mesh.bounds
                margin = 0.1 * float(np.linalg.norm(bmax - bmin))
                outside = np.any((gt_arr < bmin - margin) |
                                 (gt_arr > bmax + margin), axis=1)
                if outside.sum() > 0.25 * len(gt_arr):
                    print(f"  WARNING: {outside.sum()}/{len(gt_arr)} GT joints "
                          f"OUTSIDE mesh bbox — coordinate frame mismatch?")

            metrics = compute_all_metrics(pred["pose"], gt_skeleton, mesh)
        else:
            # FIX N1: volumetric_centering_error is now imported
            vce = volumetric_centering_error(pred["pose"], mesh)
            metrics = {"VCE": round(vce, 4) if _finite(vce) else None}

        metrics["Method"] = "Ours"
        metrics["Mesh"] = item["name"]
        metrics["Runtime_s"] = round(runtime, 2)
        all_results.append(metrics)

        

    # ---- Aggregate (FIX 7 completed) ----
    aggregate = {}
    methods = sorted(set(r.get("Method", "Ours") for r in all_results))
    for method in methods:
        method_results = [r for r in all_results if r.get("Method") == method]
        if not method_results:
            continue
        agg = {"Method": method, "N_chars": len(method_results)}
        for key in ("CD-J2J", "CD-J2B", "CD-B2B", "IoU", "Precision",
                    "Recall", "ED", "VCE", "Runtime_s"):
            # FIX 7: _finite() skips None/nan/inf — one broken mesh can
            # no longer poison the mean with inf
            vals = [r[key] for r in method_results
                    if key in r and _finite(r[key])]
            agg[f"{key}_n"] = len(vals)
            if vals:
                agg[f"{key}_mean"] = round(float(np.mean(vals)), 3)
                agg[f"{key}_std"] = round(float(np.std(vals)), 3)
        aggregate[method] = agg

    # ---- Write CSV + JSON (FIX N2: csv_path back at function level) ----
    csv_path = os.path.join(output_dir, "main_results.csv")
    with open(csv_path, "w", newline="") as f:
        if all_results:
            canonical = ["Method", "Mesh", "CD-J2J", "CD-J2B", "CD-B2B",
                         "IoU", "Precision", "Recall", "ED", "VCE", "Runtime_s"]
            fieldnames = [k for k in canonical
                          if any(k in r for r in all_results)]
            fieldnames += [k for r in all_results for k in r
                           if k not in fieldnames]
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(all_results)
    with open(os.path.join(output_dir, "main_results.json"), "w") as f:
        json.dump({"per_mesh": all_results, "aggregate": aggregate}, f, indent=2)
    print(f"\nResults written to {csv_path}")
    return aggregate

def find_outliers(results_csv_path: str):
    print("\n=== FINDING CATASTROPHIC OUTLIERS ===")
    with open(results_csv_path, 'r') as f:
        rows = list(csv.DictReader(f))

    outliers = []
    for row in rows:
        try:
            cd = float(row['CD-J2J'])
            vce = float(row['VCE']) if row['VCE'] else float('nan')
            if cd > 15.0:
                outliers.append((row['Mesh'], cd, vce))
        except (ValueError, KeyError):
            continue

    outliers.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(outliers)} catastrophic failures out of {len(rows)} characters:")
    for name, cd, vce in outliers:
        print(f"  - {name}: CD-J2J={cd:.2f}%, VCE={vce:.2f}")
        
    # Calculate the mean if we exclude the total failures
    import numpy as np
    all_cds = []
    with open(results_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cd = float(row['CD-J2J'])
                if cd < 15.0:
                    all_cds.append(cd)
            except:
                continue
                
    if all_cds:
        print(f"\nIf we exclude the catastrophic 2D-detection failures:")
        print(f"  Adjusted CD-J2J Mean: {np.mean(all_cds):.3f}%")
        print(f"  Adjusted CD-J2J Std: {np.std(all_cds):.3f}%")

# Call it at the end of main()
# find_outliers(os.path.join(args.output_dir, "main_results.csv"))

def run_cross_dataset_experiment(
    datasets: Dict[str, List[Dict]],
    output_dir: str,
) -> Dict:
    """Run cross-dataset generalisation experiment (Table VI).

    ``datasets`` is a dict mapping dataset name -> list of items.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for ds_name, items in datasets.items():
        print(f"\n=== Cross-dataset: {ds_name} ({len(items)} chars) ===")
        results[ds_name] = run_main_experiment(items, output_dir)
    with open(os.path.join(output_dir, "cross_dataset.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_ablation_experiment(
    dataset: List[Dict],
    output_dir: str,
    num_meshes: int = 7,
) -> Dict:
    """Run all 5 ablation studies across the first `num_meshes` meshes
    with GT, and aggregate mean/std per configuration across meshes.
    Replaces the old single-mesh ablation (single-mesh differences were
    within noise; cross-mesh mean±std is statistically defensible)."""
    os.makedirs(output_dir, exist_ok=True)

    subset = [it for it in dataset if it["gt_path"]][:num_meshes]
    if not subset:
        print("  No GT meshes found — skipping ablations.")
        return {}
    print(f"  Ablating on {len(subset)} meshes: "
          f"{[it['name'] for it in subset]}")

    core_joints = {
        "pelvis", "neck",
        "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
        "wrist_L", "wrist_R", "hip_L", "hip_R",
        "knee_L", "knee_R", "ankle_L", "ankle_R",
    }

    per_mesh = {}
    # ablation_name -> (label_key, label_value) -> list of metric rows
    collected: Dict[str, Dict[tuple, List[Dict]]] = {}

    for k, item in enumerate(subset):
        print(f"  [ablation {k+1}/{len(subset)}] {item['name']}")
        try:
            gt_skeleton = load_gt_skeleton(item["gt_path"])
            gt_skeleton = {n: v for n, v in gt_skeleton.items()
                           if n in core_joints}
            mesh = load_mesh_trimesh(item["mesh_path"])
            results = run_all_ablations(item["mesh_path"], gt_skeleton, mesh)
        except Exception as e:
            print(f"    Failed on {item['name']}: {e}")
            continue

        per_mesh[item["name"]] = results
        for abl_name, rows in results.items():
            groups = collected.setdefault(abl_name, {})
            for row in rows:
                # the label field identifying the variant for this ablation
                label = ("?", "?")
                for lk in ("N", "Estimator", "Config", "Method"):
                    if lk in row:
                        label = (lk, row[lk])
                        break
                groups.setdefault(label, []).append(row)

    # ---- Aggregate mean/std across meshes ----
    METRIC_KEYS = ("CD-J2J", "CD-J2B", "CD-B2B", "IoU", "Precision",
                   "Recall", "ED", "VCE", "Time_s")
    aggregate = {}
    for abl_name, groups in collected.items():
        agg_rows = []
        for (label_key, label_val), rows in groups.items():
            agg = {label_key: label_val, "n_meshes": len(rows)}
            for mk in METRIC_KEYS:
                vals = [r[mk] for r in rows if mk in r and _finite(r[mk])]
                if vals:
                    agg[f"{mk}_mean"] = round(float(np.mean(vals)), 3)
                    agg[f"{mk}_std"] = round(float(np.std(vals)), 3)
            agg_rows.append(agg)
        if abl_name == "ablation_1_num_views":
            agg_rows.sort(key=lambda r: r.get("N", 0))
        aggregate[abl_name] = agg_rows

    out = {"n_meshes": len(per_mesh), "per_mesh": per_mesh,
           "aggregate": aggregate}
    with open(os.path.join(output_dir, "ablations.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Ablations written to ablations.json "
          f"({len(per_mesh)} meshes, aggregated)")
    return out


def run_runtime_experiment(
    dataset: List[Dict],
    output_dir: str,
) -> List[Dict]:
    """Measure end-to-end runtime per character (Table XII).
    FIX #9: per-phase numbers are now MEASURED from pipeline timestamps,
    not fabricated with a 60/40 split. FIX #12: face count reuses the
    mesh the pipeline already loaded (no second load)."""
    os.makedirs(output_dir, exist_ok=True)
    pipe = VoidXPipeline()
    results = []
    for i, item in enumerate(dataset[:20]):
        try:
            t0 = time.time()
            pred = pipe.run(item["mesh_path"])
            elapsed = time.time() - t0
            md = pred["metadata"]
            results.append({
                "Mesh": item["name"],
                "Runtime_s": round(elapsed, 2),
                "Load_s": md.get("load_seconds"),
                "Phase1_s": md.get("phase1_seconds"),
                "Phase2_s": md.get("phase2_seconds"),
                "NumFaces": (len(pipe._trimesh_mesh.faces)
                             if pipe._trimesh_mesh is not None else None),
            })
        except Exception as e:
            print(f"  Failed: {e}")
    with open(os.path.join(output_dir, "runtime.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_theorem_verification(output_dir: str) -> Dict:
    """Run the Monte-Carlo verification of all 6 theorems.
    Reproduces the numbers in Tables II, IV, and Figure 4 of the paper."""
    os.makedirs(output_dir, exist_ok=True)
    results = verify_all(num_trials=2000, sigma=0.1)
    with open(os.path.join(output_dir, "theorem_verification.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VoidX experimental runner for the auto-rigging paper."
    )
    parser.add_argument("--rignet_root", type=str, default=None,
                        help="Path to RigNetv1 dataset root (with test.txt)")
    parser.add_argument("--mesh_dir", type=str, default=None,
                        help="Path to a local directory of meshes (no GT)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Where to write result JSON/CSV files")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip running external baselines (default: skip)")
    parser.add_argument("--skip_ablations", action="store_true",
                        help="Skip the ablation studies")
    parser.add_argument("--skip_theorems", action="store_true",
                        help="Skip the theorem verification step")
    args = parser.parse_args()

    # ---- Load dataset ----
    if args.rignet_root:
        dataset = load_rignet_test_split(args.rignet_root, humanoid_only=True)
        print(f"Loaded {len(dataset)} humanoid test characters from RigNetv1.")
    elif args.mesh_dir:
        dataset = load_local_meshes(args.mesh_dir)
        print(f"Loaded {len(dataset)} local meshes (no GT skeletons).")
    else:
        # Fallback: use the uploaded mesh for a smoke test
        upload_mesh = "/home/z/my-project/upload/male_t_pose0.glb"
        if os.path.exists(upload_mesh):
            dataset = [{"name": "male_t_pose0",
                        "mesh_path": upload_mesh, "gt_path": None}]
            print("No dataset specified; running smoke test on uploaded mesh.")
        else:
            print("No dataset specified and no upload mesh found. Exiting.")
            return

    # ---- Step 1: Theorem verification ----
    if not args.skip_theorems:
        print("\n=== Step 1: Theorem verification (Monte Carlo) ===")
        run_theorem_verification(args.output_dir)

    # ---- Step 2: Main experiment ----
    print("\n=== Step 2: Main quantitative experiment (Table V) ===")
    aggregate = run_main_experiment(dataset, args.output_dir)
    print("\nAggregate results:")
    for method, agg in aggregate.items():
        print(f"  {method}: {agg}")

    # ---- Step 3: Ablations (across 07 meshes with GT) ----
    if not args.skip_ablations and dataset:
        print("\n=== Step 3: Ablation studies (multi-mesh) ===")
        run_ablation_experiment(dataset, args.output_dir, num_meshes=7)
    # ---- Step 4: Runtime ----
    print("\n=== Step 4: Runtime analysis (Table XII) ===")
    run_runtime_experiment(dataset, args.output_dir)

    print(f"\nAll experiments complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
