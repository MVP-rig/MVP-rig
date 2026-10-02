

# MVP-Rig — Multi-View Pose Rigging

[![DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.22946123-blue)](https://doi.org/10.5281/zenodo.22946123)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)

 [Example.webm](https://github.com/user-attachments/assets/e7130336-2682-4382-9a6c-953cdf2357d4)


**MVP-Rig is a deterministic, training-free pipeline that rigs a humanoid mesh with a 14-joint skeleton in ~6 seconds — no training, no datasets, no learned weights.**

It renders the mesh from 8 near-orthographic viewpoints, detects 2D pose landmarks with MediaPipe Holistic, lifts them to 3D with closed-form trigonometry, then anchors every joint to the *volumetric center* of its limb using raycasting.


mesh.obj ──▶ 8 orthographic renders ──▶ MediaPipe 2D landmarks
          ──▶ Phase 1: multi-view 3D lifting (closed-form trig)
          ──▶ Phase 2: raycast volumetric centering
          ──▶ 14-joint skeleton (JSON)


## Results at a glance

Evaluated on a 71-character humanoid test set (see [Dataset](#dataset-format)):

| Metric | Value |
|---|---|
| CD-J2J (% of bbox diagonal) | **2.35 ± 0.46** |
| CD-J2B | 1.45 ± 0.43 |
| CD-B2B | 1.27 ± 0.37 |
| IoU | 95.5 ± 9.8% |
| VCE | 0.207 ± 0.060 |
| Runtime per character | 6.1 ± 3.7 s |

*Under a matched 14-joint protocol on the authors' own dataset — not directly comparable to full-skeleton RigNet-protocol numbers.*

Monte-Carlo verification of the estimation-theory analysis (CRLB for multi-view depth) matches theory — see `results/theorem_verification.json`.

---

## Installation

Tested on **Windows 11, Python 3.11**.

```bash
git clone https://github.com/KRISH9621/MVP-rig.git
cd MVP-rig
pip install numpy scipy trimesh open3d mediapipe opencv-python plotly
```

> MediaPipe and Open3D are **required** — the pipeline fails fast if either is missing.

---

## Dataset format

<dataset_root>/
├── char001/
│   ├── mesh.obj          # humanoid mesh (T-pose or A-pose)
│   └── bone_3d.json      # reference skeleton (optional)
├── char002/
│   ├── mesh.obj
│   └── bone_3d.json
└── ...


- **Mesh:** `.obj` (`.glb` also handled). One mesh per subfolder.
- **Reference skeleton** `bone_3d.json`: flat JSON dict `{"JointName": [x, y, z], ...}`. Mixamo names are mapped automatically.
- **Download the test set:** [![DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.22946123-blue)](https://doi.org/10.5281/zenodo.22946123)

---

## Usage

### 1. Rig a single mesh

```bash
python voidx_pipeline.py path/to/char001/mesh.obj
# → skeleton JSON output
```

### 2. Full evaluation suite

```bash
python voidx_experiment.py --mesh_dir path/to/dataset --output_dir ./results
```

Runs: theorem verification → main experiment → ablations → runtime profiling.

### 3. Outputs

| File | Contents |
|---|---|
| `results/main_results.csv` / `.json` | Per-character metrics + aggregate |
| `results/preds/*.json` | **Frozen per-character predictions** |
| `results/ablations.json` | Ablation tables (per-mesh + aggregated) |
| `results/theorem_verification.json` | CRLB verification tables |

### 4. Visualization

```bash
python view_rig.py results/preds/char002.json    # one character
python view_rig.py --batch results/preds          # all characters
python view_rig.py --worst 3                       # failure cases
```

---

## Known limitations

We state these explicitly:

1. **Bulky accessories** (pocket boxes, bags) can pull the volumetric center away from the body's true limb center.
2. **Joint-line confusion**: when an upper-limb joint falls in the same vertical line as a lower-limb joint, the vertical scaling can misplace it.
3. **Thick layered clothing** obscures the body's internal structure, preventing accurate limb center recovery.
4. **Complex non-planar poses** (crossed arms, raised legs) degrade accuracy as the pose deviates from the fully extended A/T-pose.

None of these limitations is hidden by post-filtering: every character is included in every reported mean.

---

## Repository layout

```
voidx_pipeline.py       # Phase 1 + Phase 2 pipeline (the method)
voidx_theorems.py       # Estimation theory + Monte-Carlo verification
voidx_metrics.py        # CD-*, IoU, VCE metrics
voidx_experiment.py     # Full evaluation suite
voidx_ablation.py       # 5 ablation studies
view_rig.py           # Interactive 3D QA viewer
```

---

## Citation

```bibtex
@software{Maurya2026mvprig,
  title={Deterministic Volumetric Auto-Rigging via Multi-View
         Orthographic Lifting of 2D Pose Landmarks},
  author={Maurya, Krishnanand and Agostino, Christopher J.},
  year={2026},
  url={https://github.com/KRISH9621/MVP-rig},
  version={1.0.0}
}
```

## 📜License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Email: nanddynasty5@gmail.com
GitHub: [@KRISH9621](https://github.com/KRISH9621/MVP-rig)

