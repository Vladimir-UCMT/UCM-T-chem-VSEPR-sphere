# viz_points_sphere.py
# Visualize VSEPR-on-sphere runs from *_results_global.json
#
# Outputs:
#   <tag>__viz.png   (always)
#   <tag>__viz.html  (optional, if --html; requires plotly)
#   <tag>__spin.gif  (optional, if --gif; requires imageio)
#
# Examples:
#   python viz_points_sphere.py results\PURE_N8_a1p10_results_global.json --labels --knn 4 --contrast --gif
#   python viz_points_sphere.py results\PURE_N12_a1p10_results_global.json --knn 3 --contrast --gif --gif-seconds 7
#   python viz_points_sphere.py results\N6_NB2_NL4_scan_a2p1_results_global.json --labels --knn 3 --edges-by-type --contrast --gif
#   python viz_points_sphere.py results\PURE_N12_a1p10_results_global.json --html --knn 3
#
# Requires: numpy, matplotlib
# Optional: plotly (for --html), imageio (for --gif)

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def load_run_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_nb_nl(types_list):
    nb = sum(1 for t in types_list if t == "B")
    nl = sum(1 for t in types_list if t == "L")
    return nb, nl


def draw_sphere_wire(ax, r=1.0, n=26, alpha=0.20, lw=0.4):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=lw, alpha=alpha)


def angle_deg(u, v):
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def compute_knn_edges(points_xyz: np.ndarray, k: int):
    """Undirected kNN edges; returns list of (i,j) with i<j."""
    n = points_xyz.shape[0]
    edges = set()
    for i in range(n):
        angs = []
        for j in range(n):
            if i == j:
                continue
            angs.append((angle_deg(points_xyz[i], points_xyz[j]), j))
        angs.sort(key=lambda t: t[0])
        for _, j in angs[:k]:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return sorted(edges)


def edge_type(ti, tj):
    if ti == "B" and tj == "B":
        return "BB"
    if ti == "L" and tj == "L":
        return "LL"
    return "BL"


def plot_matplotlib(
    points,
    types_list,
    title,
    out_png,
    labels=False,
    show_sphere=True,
    knn=0,
    edges_by_type=False,
    contrast=False,
    azim=None,
    save_path=None,
):
    pts = points

    # Contrast tuning
    pt_size_b = 60 if not contrast else 105
    pt_size_l = 65 if not contrast else 120
    edge_alpha = 0.25 if not contrast else 0.60
    sphere_alpha = 0.20 if not contrast else 0.10
    sphere_lw = 0.4 if not contrast else 0.25
    label_fs = 9 if not contrast else 11

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    if show_sphere:
        draw_sphere_wire(ax, r=1.0, n=26, alpha=sphere_alpha, lw=sphere_lw)

    idx_b = [i for i, t in enumerate(types_list) if t == "B"]
    idx_l = [i for i, t in enumerate(types_list) if t == "L"]

    if idx_b:
        pb = pts[idx_b]
        ax.scatter(pb[:, 0], pb[:, 1], pb[:, 2], s=pt_size_b, marker="o", label="B")
    if idx_l:
        pl = pts[idx_l]
        ax.scatter(pl[:, 0], pl[:, 1], pl[:, 2], s=pt_size_l, marker="^", label="L")

    # Edges (kNN)
    if knn and knn > 0:
        edges = compute_knn_edges(pts, knn)
        for i, j in edges:
            x = [pts[i, 0], pts[j, 0]]
            y = [pts[i, 1], pts[j, 1]]
            z = [pts[i, 2], pts[j, 2]]

            if edges_by_type:
                et = edge_type(types_list[i], types_list[j])
                if et == "BB":
                    lw, a = 2.2, edge_alpha * 1.0
                elif et == "LL":
                    lw, a = 1.8, edge_alpha * 0.55
                else:  # BL
                    lw, a = 1.4, edge_alpha * 0.40
                ax.plot(x, y, z, linewidth=lw, alpha=a)
            else:
                ax.plot(x, y, z, linewidth=1.4 if contrast else 1.0, alpha=edge_alpha)

    if labels:
        for i, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, f"{i}", fontsize=label_fs)

    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")

    if azim is not None:
        ax.view_init(elev=20, azim=float(azim))

    plt.tight_layout()
    target = save_path if save_path is not None else out_png
    plt.savefig(target, dpi=220)
    plt.close(fig)


def plot_plotly_html(points, types_list, title, out_html, knn=0, edges_by_type=False):
    try:
        import plotly.graph_objects as go
    except Exception:
        raise RuntimeError("Plotly not installed. Install: python -m pip install plotly")

    pts = points
    idx_b = [i for i, t in enumerate(types_list) if t == "B"]
    idx_l = [i for i, t in enumerate(types_list) if t == "L"]

    traces = []

    if idx_b:
        pb = pts[idx_b]
        traces.append(
            go.Scatter3d(
                x=pb[:, 0],
                y=pb[:, 1],
                z=pb[:, 2],
                mode="markers+text",
                text=[str(i) for i in idx_b],
                textposition="top center",
                marker=dict(size=5),
                name="B",
            )
        )

    if idx_l:
        pl = pts[idx_l]
        traces.append(
            go.Scatter3d(
                x=pl[:, 0],
                y=pl[:, 1],
                z=pl[:, 2],
                mode="markers+text",
                text=[str(i) for i in idx_l],
                textposition="top center",
                marker=dict(size=6, symbol="diamond"),
                name="L",
            )
        )

    if knn and knn > 0:
        edges = compute_knn_edges(pts, knn)
        for i, j in edges:
            et = edge_type(types_list[i], types_list[j])
            xs = [pts[i, 0], pts[j, 0], None]
            ys = [pts[i, 1], pts[j, 1], None]
            zs = [pts[i, 2], pts[j, 2], None]

            width = 2
            opacity = 0.35
            if edges_by_type:
                if et == "BB":
                    width, opacity = 4, 0.55
                elif et == "LL":
                    width, opacity = 3, 0.30
                else:
                    width, opacity = 2, 0.22

            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(width=width),
                    opacity=opacity,
                    showlegend=False,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.15, 1.15]),
            yaxis=dict(range=[-1.15, 1.15]),
            zaxis=dict(range=[-1.15, 1.15]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_html))


def save_spin_gif(make_frame_fn, out_gif: Path, frames=48, duration=6):
    """make_frame_fn(azim_deg, frame_path) -> frame_path"""
    if imageio is None:
        raise RuntimeError("imageio not installed. Install: python -m pip install imageio")

    imgs = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for k in range(frames):
            az = 360.0 * k / frames
            frame_path = td / f"frame_{k:03d}.png"
            png = make_frame_fn(az, frame_path)
            imgs.append(imageio.imread(png))
        imageio.mimsave(out_gif, imgs, duration=(duration / frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to *_results_global.json")
    ap.add_argument("--outdir", default="", help="Output directory (default: same as json)")
    ap.add_argument("--labels", action="store_true", help="Draw point index labels on PNG")
    ap.add_argument("--no_sphere", action="store_true", help="Disable sphere wireframe")
    ap.add_argument("--knn", type=int, default=0, help="Draw edges to k nearest neighbors (e.g. 3 or 4)")
    ap.add_argument("--edges-by-type", action="store_true", help="Vary edge style for BB/BL/LL if types exist")
    ap.add_argument("--html", action="store_true", help="Also save interactive HTML (requires plotly)")
    ap.add_argument("--contrast", action="store_true", help="Increase visual contrast (bigger points/thicker edges)")
    ap.add_argument("--gif", action="store_true", help="Also save rotating GIF (requires imageio)")
    ap.add_argument("--gif-frames", type=int, default=48, help="Frames in GIF (default 48)")
    ap.add_argument("--gif-seconds", type=int, default=6, help="Total GIF duration in seconds (default 6)")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    data = load_run_json(json_path)

    pts = np.array(data["points_xyz"], dtype=float)
    types_list = data.get("types", ["B"] * len(pts))
    tag = data.get("tag", json_path.stem)
    alpha = data.get("alpha", None)
    energy = data.get("energy", None)
    N = int(data.get("N", len(pts)))
    nb, nl = infer_nb_nl(types_list)

    outdir = Path(args.outdir) if args.outdir else json_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    out_png = outdir / f"{tag}__viz.png"
    out_html = outdir / f"{tag}__viz.html"
    out_gif = outdir / f"{tag}__spin.gif"

    title = f"{tag} | N={N} (NB={nb}, NL={nl})"
    if alpha is not None:
        title += f" | alpha={alpha}"
    if energy is not None:
        try:
            title += f" | E={float(energy):.6g}"
        except Exception:
            pass

    plot_matplotlib(
        points=pts,
        types_list=types_list,
        title=title,
        out_png=out_png,
        labels=args.labels,
        show_sphere=not args.no_sphere,
        knn=args.knn,
        edges_by_type=args.edges_by_type,
        contrast=args.contrast,
        azim=None,
        save_path=None,
    )
    print(f"Saved PNG: {out_png}")

    if args.html:
        plot_plotly_html(
            points=pts,
            types_list=types_list,
            title=title,
            out_html=out_html,
            knn=args.knn,
            edges_by_type=args.edges_by_type,
        )
        print(f"Saved HTML: {out_html}")

    if args.gif:
        def make_frame(azim, frame_path):
            plot_matplotlib(
                points=pts,
                types_list=types_list,
                title=title,
                out_png=out_png,
                labels=args.labels,
                show_sphere=not args.no_sphere,
                knn=args.knn,
                edges_by_type=args.edges_by_type,
                contrast=args.contrast,
                azim=azim,
                save_path=frame_path,
            )
            return frame_path

        save_spin_gif(
            make_frame_fn=make_frame,
            out_gif=out_gif,
            frames=args.gif_frames,
            duration=args.gif_seconds,
        )
        print(f"Saved GIF: {out_gif}")


if __name__ == "__main__":
    main()