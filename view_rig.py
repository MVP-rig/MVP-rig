"""
view_rig.py — final visual QA + paper-figure generator.

Usage:
    python view_rig.py results\preds\char2.json
    python view_rig.py --batch results\preds
    python view_rig.py --worst 5
    python view_rig.py --pred-only results\preds\char2.json
        (mesh + YOUR skeleton only — no reference, no error lines)

Legend (click any item to show/hide the entire group):
    pale blue = mesh | RED dots + GREEN bones = prediction
    BLUE diamonds + blue bones = reference (Mixamo)
    CYAN x = cross-section centers | ORANGE dashed = error connectors
"""
import sys, os, json, csv, argparse
import numpy as np
import trimesh
import plotly.graph_objects as go

BONE_CONNECTIONS = [
    ("pelvis", "hip_L"), ("pelvis", "hip_R"),
    ("hip_L", "knee_L"), ("hip_R", "knee_R"),
    ("knee_L", "ankle_L"), ("knee_R", "ankle_R"),
    ("pelvis", "neck"),
    ("neck", "shoulder_L"), ("neck", "shoulder_R"),
    ("shoulder_L", "elbow_L"), ("shoulder_R", "elbow_R"),
    ("elbow_L", "wrist_L"), ("elbow_R", "wrist_R"),
]
MIXAMO_MAP = {
    "Hips": "pelvis", "Neck": "neck",
    "LeftShoulder": "shoulder_L", "RightShoulder": "shoulder_R",
    "LeftForeArm": "elbow_L", "RightForeArm": "elbow_R",
    "LeftHand": "wrist_L", "RightHand": "wrist_R",
    "LeftUpLeg": "hip_L", "RightUpLeg": "hip_R",
    "LeftLeg": "knee_L", "RightLeg": "knee_R",
    "LeftFoot": "ankle_L", "RightFoot": "ankle_R",
}
CORE = {n for pair in BONE_CONNECTIONS for n in pair} | {"pelvis", "neck"}
JUNCTION_JOINTS = {"shoulder_L", "shoulder_R", "hip_L", "hip_R",
                   "pelvis", "neck"}


def load_mesh(path):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        geoms = []
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph.get(node)
            g = scene.geometry[gname].copy()
            g.apply_transform(T)
            geoms.append(g)
        return trimesh.util.concatenate(geoms)
    return scene


def map_name(jn):
    clean = jn.split(":")[-1].strip()
    for m, i in MIXAMO_MAP.items():
        if m.lower() == clean.lower():
            return i
    return clean


def load_gt(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("joints", "bones", "skeleton", "pose"):
            if key in raw and isinstance(raw[key], (dict, list)):
                raw = raw[key]; break
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, (list, tuple)) and len(v) == 3:
                out[map_name(k)] = np.array(v, dtype=float)
            elif isinstance(v, dict) and all(c in v for c in "xyz"):
                out[map_name(k)] = np.array([v["x"], v["y"], v["z"]], dtype=float)
    return {k: v for k, v in out.items() if k in CORE}


def seg_dist(p, a, b):
    ab = b - a; ap = p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def joint_centering(pos, bones, mesh):
    best, best_d = None, np.inf
    for a, b in bones:
        d = seg_dist(pos, a, b)
        if d < best_d:
            best_d, best = d, (a, b)
    if best is None:
        return None, None
    a, b = best
    ab = b - a
    t = np.clip(np.dot(pos - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    proj = a + t * ab
    bd = ab / max(np.linalg.norm(ab), 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(bd[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(bd, ref); u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(bd, u)
    hits = {}
    for k in range(8):
        ang = 2 * np.pi * k / 8
        d = np.cos(ang) * u + np.sin(ang) * v
        try:
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=np.array([proj]), ray_directions=np.array([d]))
            if len(locs) > 0:
                hits[k] = np.asarray(locs[0])
        except Exception:
            pass
    mids, pairs = [], []
    for k in range(4):
        if k in hits and (k + 4) in hits:
            mids.append(0.5 * (hits[k] + hits[k + 4]))
            pairs.append(float(np.linalg.norm(hits[k] - hits[k + 4])))
    if len(pairs) < 2 or np.median(pairs) < 1e-6:
        return None, None
    center = np.mean(mids, axis=0)
    ev = pos - center
    radial = ev - np.dot(ev, bd) * bd
    return float(np.linalg.norm(radial) / np.median(pairs)), center


def load_metrics_row(csv_path, mesh_name):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("Mesh") == mesh_name:
                    return row
    except Exception:
        pass
    return None


def build_figure(mesh_path, pred_path, gt_path=None, metrics_row=None,
                 mesh_opacity=0.15, out_dir=".", pred_only=False):
    with open(pred_path) as f:
        dna = json.load(f)
    if not mesh_path or not os.path.exists(mesh_path):
        mesh_path = dna.get("metadata", {}).get("mesh_path", "")
    if not mesh_path or not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found for {pred_path}")
    if gt_path is None:
        gt_path = os.path.join(os.path.dirname(mesh_path), "bone_3d.json")

    mesh = load_mesh(mesh_path)
    pred = {k: np.array([v["x"], v["y"], v["z"]])
            for k, v in dna["pose"].items()
            if isinstance(v, dict) and "x" in v}
    gt = load_gt(gt_path)
    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    char_name = os.path.splitext(os.path.basename(pred_path))[0]

    bones = [(pred[a], pred[b]) for a, b in BONE_CONNECTIONS
             if a in pred and b in pred]

    fig = go.Figure()

    # ---- mesh (group: mesh) ----
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        opacity=mesh_opacity, color="#7FD4FF", name="Mesh",
        hoverinfo="skip", legendgroup="mesh"))

    # ---- reference skeleton (group: ref) — SKIPPED in pred-only mode ----
    if gt and not pred_only:
        gx = [p[0] for p in gt.values()]; gy = [p[1] for p in gt.values()]
        gz = [p[2] for p in gt.values()]
        fig.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz, mode="markers",
            marker=dict(size=4, color="#1E90FF", symbol="diamond"),
            name="Reference skeleton",
            text=[f"GT {n}" for n in gt],
            hoverinfo="text", legendgroup="ref"))
        for a, b in BONE_CONNECTIONS:
            if a in gt and b in gt:
                fig.add_trace(go.Scatter3d(
                    x=[gt[a][0], gt[b][0]], y=[gt[a][1], gt[b][1]],
                    z=[gt[a][2], gt[b][2]], mode="lines",
                    line=dict(color="#1E90FF", width=4),
                    name=f"ref: {a}-{b}", hoverinfo="skip",
                    legendgroup="ref", showlegend=False))
    elif not pred_only:
        print(f"  NOTE: no reference found at {gt_path}")

    # ---- predicted joints + bones (group: pred) ----
    px, py, pz, ptext = [], [], [], []
    cx, cy, cz, ex, ey, ez = [], [], [], [], [], []
    for n, p in pred.items():
        px.append(p[0]); py.append(p[1]); pz.append(p[2])
        hover = f"<b>{n}</b>"
        if n in JUNCTION_JOINTS:
            err, center = None, None
        else:
            err, center = joint_centering(p, bones, mesh)
        if err is not None:
            hover += f"<br>radial centering err: {err:.3f} x diameter"
            cx.append(center[0]); cy.append(center[1]); cz.append(center[2])
        if n in gt and not pred_only:
            d = float(np.linalg.norm(p - gt[n]))
            verdict = "PASS" if d <= 0.05 * diag else "fail"
            hover += (f"<br>dist to ref: {d:.4f} ({100*d/diag:.2f}% diag)"
                      f" <b>[{verdict} @ tau=5%]</b>")
            ex += [p[0], gt[n][0], None]
            ey += [p[1], gt[n][1], None]
            ez += [p[2], gt[n][2], None]
        ptext.append(hover)

    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, mode="markers+text",
        text=list(pred.keys()), textposition="top center",
        textfont=dict(size=8), marker=dict(size=5, color="red"),
        name="Predicted joints", hovertext=ptext, hoverinfo="text",
        legendgroup="pred"))

    for a, b in BONE_CONNECTIONS:
        if a in pred and b in pred:
            fig.add_trace(go.Scatter3d(
                x=[pred[a][0], pred[b][0]], y=[pred[a][1], pred[b][1]],
                z=[pred[a][2], pred[b][2]], mode="lines",
                line=dict(color="#32CD32", width=6),
                name=f"bone: {a}-{b}", hoverinfo="skip",
                legendgroup="pred", showlegend=False))

    # ---- cross-section centers (group: centers) — pred-only skips ----
    if cx and not pred_only:
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode="markers",
            marker=dict(size=3, color="#00FFFF", symbol="x"),
            name="Cross-section centers", hoverinfo="skip",
            legendgroup="centers"))

    # ---- error connectors (group: error) — pred-only skips ----
    if ex and not pred_only:
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="#FFA500", width=2, dash="dot"),
            name="Pred-ref error", hoverinfo="skip",
            legendgroup="error"))

    title = f"Rig QA - {char_name}"
    if metrics_row:
        title += (f" | CD-J2J {metrics_row.get('CD-J2J', '?')}%"
                  f" | PCK@5% {metrics_row.get('Precision', '?')}%"
                  f" | VCE {metrics_row.get('VCE', '?')}")

    out = os.path.join(out_dir, f"rig_qa_{char_name}.html")
    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data",
                   camera=dict(eye=dict(x=1.7, y=1.7, z=0.9))),
        margin=dict(l=0, r=0, b=0, t=50), template="plotly_dark")
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def main():
    ap = argparse.ArgumentParser(description="MVP-Rig QA viewer")
    ap.add_argument("files", nargs="*", help="pred json, or mesh.obj + pred.json [+ gt]")
    ap.add_argument("--batch", metavar="PREDS_DIR", help="render every pred in dir")
    ap.add_argument("--worst", type=int, metavar="N",
                    help="render the N worst characters by CD-J2J")
    ap.add_argument("--pred-only", action="store_true",
                    help="mesh + predicted skeleton only (no reference, no error lines)")
    ap.add_argument("--results", default=r".\results", help="results dir")
    ap.add_argument("--out", default=".", help="output dir for HTML files")
    ap.add_argument("--opacity", type=float, default=0.15,
                    help="mesh opacity (0.15 for QA, 0.35 for pretty figures)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.results, "main_results.csv")

    jobs = []
    if args.worst:
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found.")
            sys.exit(1)
        rows, skipped = [], 0
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((float(r["CD-J2J"]), str(r["Mesh"]).strip()))
                except (ValueError, KeyError, TypeError):
                    skipped += 1
        if not rows:
            print(f"ERROR: parsed 0 rows from {csv_path}")
            sys.exit(1)
        rows.sort(reverse=True)
        for _, name in rows[:args.worst]:
            p = os.path.join(args.results, "preds", f"{name}.json")
            if os.path.exists(p):
                jobs.append((None, p, None, name))
    if args.batch:
        for f in sorted(os.listdir(args.batch)):
            if f.endswith(".json"):
                jobs.append((None, os.path.join(args.batch, f), None, f[:-5]))
    if args.files:
        if len(args.files) == 1:
            jobs.append((None, args.files[0], None))
        else:
            jobs.append((args.files[0], args.files[1],
                         args.files[2] if len(args.files) > 2 else None))

    if not jobs:
        print(__doc__); sys.exit(1)

    for job in jobs:
        mesh_path, pred_path, gt_path = job[0], job[1], job[2]
        name = job[3] if len(job) > 3 else \
            os.path.splitext(os.path.basename(pred_path))[0]
        try:
            row = load_metrics_row(csv_path, name)
            out = build_figure(mesh_path, pred_path, gt_path, row,
                               mesh_opacity=args.opacity,
                               out_dir=args.out, pred_only=args.pred_only)
            print(f"  OK {out}")
        except Exception as e:
            print(f"  FAIL {pred_path}: {e}")


if __name__ == "__main__":
    main()