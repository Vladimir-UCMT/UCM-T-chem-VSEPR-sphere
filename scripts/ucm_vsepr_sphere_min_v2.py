# ucm_vsepr_sphere_min_v2.py  (v2.1)
# Small update: prints a single "bond angle" line for easy copy into a paper table.
#
# Bond angle definition:
#   - if NB >= 2: bond angle = median of BB angles (in degrees)
#   - if NB == 2: it's the single BB angle
#   - if NB < 2: not defined
#
# Everything else is identical to your v2.

import argparse
import json
import math
import os
import random
from typing import List, Tuple, Dict

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1e-12, None)
    return X / nrm


def random_points_on_sphere(N: int) -> np.ndarray:
    X = np.random.normal(size=(N, 3))
    return normalize_rows(X)


def angle_degrees(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.dot(u, v))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def pairwise_angles(X: np.ndarray) -> List[float]:
    N = X.shape[0]
    ang = []
    for i in range(N):
        for j in range(i + 1, N):
            ang.append(angle_degrees(X[i], X[j]))
    return ang


def stats_angles(angles: List[float]) -> Dict[str, float]:
    a = np.array(angles, dtype=float)
    return {
        "count": int(a.size),
        "min_deg": float(a.min()),
        "max_deg": float(a.max()),
        "mean_deg": float(a.mean()),
        "p10_deg": float(np.percentile(a, 10)),
        "p50_deg": float(np.percentile(a, 50)),
        "p90_deg": float(np.percentile(a, 90)),
        "std_deg": float(a.std()),
    }


def energy_repulsion(X: np.ndarray, w: np.ndarray) -> float:
    N = X.shape[0]
    E = 0.0
    for i in range(N):
        xi = X[i]
        for j in range(i + 1, N):
            c = float(np.dot(xi, X[j]))
            c = max(-1.0, min(1.0, c))
            denom = 1.0 - c
            if denom < 1e-9:
                denom = 1e-9
            E += float(w[i, j]) / denom
    return E


def build_types(NB: int, NL: int) -> List[str]:
    return ["B"] * NB + ["L"] * NL


def build_weights_from_alpha(types: List[str], alpha: float) -> np.ndarray:
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.0")
    N = len(types)
    w = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            ti, tj = types[i], types[j]
            if ti == "B" and tj == "B":
                wij = 1.0
            elif (ti == "B" and tj == "L") or (ti == "L" and tj == "B"):
                wij = float(alpha)
            else:
                wij = float(alpha) * float(alpha)
            w[i, j] = w[j, i] = wij
    return w


def local_step_projected(X: np.ndarray, w: np.ndarray, step: float) -> np.ndarray:
    N = X.shape[0]
    G = np.zeros_like(X)
    for i in range(N):
        xi = X[i]
        gi = np.zeros(3, dtype=float)
        for j in range(N):
            if j == i:
                continue
            xj = X[j]
            c = float(np.dot(xi, xj))
            c = max(-1.0, min(1.0, c))
            denom = 1.0 - c
            denom = max(denom, 1e-6)
            gi += float(w[i, j]) * xj / (denom * denom)
        G[i] = -gi

    Xn = X + step * G
    Xn = normalize_rows(Xn)
    return Xn


def anneal_optimize(
    N: int,
    w: np.ndarray,
    restarts: int = 80,
    iters: int = 3000,
    step0: float = 0.02,
    step_min: float = 0.002,
    noise0: float = 0.08,
    noise_min: float = 0.005,
    seed: int = 0,
) -> Tuple[np.ndarray, float, Dict]:
    set_seed(seed)

    best_E = float("inf")
    best_X = None
    energies = []

    for _ in range(restarts):
        X = random_points_on_sphere(N)
        E = energy_repulsion(X, w)

        for t in range(iters):
            frac = t / max(1, iters - 1)
            step = step0 * (1.0 - frac) + step_min * frac
            noise = noise0 * (1.0 - frac) + noise_min * frac

            Xp = local_step_projected(X, w, step)
            Xp = Xp + noise * np.random.normal(size=Xp.shape)
            Xp = normalize_rows(Xp)

            Ep = energy_repulsion(Xp, w)
            if Ep < E:
                X, E = Xp, Ep

        energies.append(E)
        if E < best_E:
            best_E = E
            best_X = X.copy()

    arr = np.array(energies, dtype=float)
    diag = {
        "restarts": int(restarts),
        "iters": int(iters),
        "seed": int(seed),
        "energy_best": float(best_E),
        "energy_median": float(np.median(arr)),
        "energy_min": float(arr.min()),
        "energy_max": float(arr.max()),
    }
    return best_X, best_E, diag


def angle_lists_by_pairtype(X: np.ndarray, types: List[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"BB": [], "BL": [], "LL": []}
    N = X.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            a = angle_degrees(X[i], X[j])
            t = types[i] + types[j]
            if t == "BB":
                out["BB"].append(a)
            elif t in ("BL", "LB"):
                out["BL"].append(a)
            else:
                out["LL"].append(a)
    return out

def bond_angle_line(case_name: str, NB: int, NL: int, alpha: float, by: Dict[str, List[float]]) -> str:
    """
    For papers: print characteristic bond angles for BB.
    - If BB angles exist: report p25, p50, max (helps for multi-angle shapes like AX3E2, AX4E).
    """
    ax_label = f"AX{NB}" + (f"E{NL}" if NL > 0 else "")
    if NB < 2 or len(by["BB"]) == 0:
        return f"{case_name} | {ax_label} | alpha={alpha:.3f} | BB: n/a"

    bb = np.array(by["BB"], dtype=float)
    p25 = float(np.percentile(bb, 25))
    p50 = float(np.percentile(bb, 50))
    p75 = float(np.percentile(bb, 75))
    mx = float(bb.max())

    # If it's essentially a single angle (like CH4), p25~p50~p75~max
    return (
        f"{case_name} | {ax_label} | alpha={alpha:.3f} | "
        f"BB(p25,p50,p75,max)=({p25:.3f}°, {p50:.3f}°, {p75:.3f}°, {mx:.3f}°)"
    )

def classify_basic(N: int, types: List[str], X: np.ndarray) -> str:
    by = angle_lists_by_pairtype(X, types)
    msg = []

    NB = types.count("B")
    NL = types.count("L")
    msg.append(f"types: NB={NB}, NL={NL}")

    if NL == 0:
        a = np.array(pairwise_angles(X), dtype=float)

        def count_near(x, tol):
            return int(np.sum(np.abs(a - x) <= tol))

        if N == 5:
            c90 = count_near(90.0, 6.0)
            c120 = count_near(120.0, 6.0)
            c180 = count_near(180.0, 6.0)
            msg.append(f"N=5 heuristic: near90={c90}, near120={c120}, near180={c180}. TBP expected ~ (6,3,1).")
        elif N == 6:
            c90 = count_near(90.0, 6.0)
            c180 = count_near(180.0, 6.0)
            msg.append(f"N=6 heuristic: near90={c90}, near180={c180}. Octa expected ~ (12,3).")
        else:
            msg.append("pure-B case: no built-in classifier for this N.")
        return " ".join(msg)

    if len(by["BB"]) > 0:
        bb = np.array(by["BB"], dtype=float)
        msg.append(f"BB angles: min={bb.min():.3f}, p50={np.percentile(bb,50):.3f}, max={bb.max():.3f}")
    if len(by["BL"]) > 0:
        bl = np.array(by["BL"], dtype=float)
        msg.append(f"BL angles: min={bl.min():.3f}, p50={np.percentile(bl,50):.3f}, max={bl.max():.3f}")
    if len(by["LL"]) > 0:
        ll = np.array(by["LL"], dtype=float)
        msg.append(f"LL angles: min={ll.min():.3f}, p50={np.percentile(ll,50):.3f}, max={ll.max():.3f}")

    if NB == 3 and NL == 1 and N == 4:
        msg.append("case hint: NH3-like (AX3E). Expect BB median < 109.47 (often ~107).")
    if NB == 2 and NL == 2 and N == 4:
        msg.append("case hint: H2O-like (AX2E2). Expect BB median < NH3 (often ~104-105).")

    return " ".join(msg)


def write_results(out_dir: str, tag: str, types: List[str], alpha: float, X: np.ndarray, E: float, diag: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)

    all_ang = pairwise_angles(X)
    s_all = stats_angles(all_ang)
    by = angle_lists_by_pairtype(X, types)
    s_by = {k: (stats_angles(v) if len(v) else None) for k, v in by.items()}

    js = {
        "tag": tag,
        "N": int(X.shape[0]),
        "types": types,
        "alpha": float(alpha),
        "energy": float(E),
        "diag": diag,
        "angles_stats_all": s_all,
        "angles_stats_by_type": s_by,
        "angles_all_deg": [float(a) for a in all_ang],
        "points_xyz": X.tolist(),
    }
    with open(os.path.join(out_dir, f"{tag}_results_global.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, f"{tag}_results_items.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("i,j,type_pair,weight,angle_deg\n")
        N = X.shape[0]
        w = build_weights_from_alpha(types, alpha)
        for i in range(N):
            for j in range(i + 1, N):
                tp = types[i] + types[j]
                if tp == "LB":
                    tp = "BL"
                a = angle_degrees(X[i], X[j])
                f.write(f"{i},{j},{tp},{w[i,j]:.6f},{a:.6f}\n")

    verdict = classify_basic(X.shape[0], types, X)
    with open(os.path.join(out_dir, f"{tag}_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(verdict + "\n")


def parse_case(case: str) -> Tuple[str, int, int]:
    c = (case or "").strip().lower()
    if c in ("nh3", "ax3e"):
        return "NH3", 3, 1
    if c in ("h2o", "ax2e2"):
        return "H2O", 2, 2
    if c in ("ch4", "ax4"):
        return "CH4", 4, 0
    if c in ("co2", "ax2"):
        return "CO2", 2, 0
    raise ValueError("Unknown case. Use NH3, H2O, CH4, CO2 or specify --NB/--NL explicitly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="", help="Convenience: NH3, H2O, CH4, CO2")
    ap.add_argument("--NB", type=int, default=None, help="Number of B directions")
    ap.add_argument("--NL", type=int, default=None, help="Number of L directions")
    ap.add_argument("--alpha", type=float, default=1.2, help="alpha = A_L/A_B (>1). Sets weights 1, alpha, alpha^2")
    ap.add_argument("--restarts", type=int, default=80)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    alpha = float(args.alpha)
    if alpha <= 1.0:
        raise SystemExit("alpha must be > 1.0 (try 1.1..1.6)")

    case_name = ""
    if args.case:
        case_name, NB, NL = parse_case(args.case)
    else:
        if args.NB is None or args.NL is None:
            raise SystemExit("Provide either --case (NH3/H2O/CH4/CO2) or both --NB and --NL.")
        NB, NL = int(args.NB), int(args.NL)
        case_name = f"NB{NB}_NL{NL}"

    if NB < 0 or NL < 0:
        raise SystemExit("NB and NL must be >= 0")
    N = NB + NL
    if N < 2:
        raise SystemExit("Total N=NB+NL must be >= 2")

    types = build_types(NB, NL)
    w = build_weights_from_alpha(types, alpha)

    tag = args.tag.strip() or f"N{N}_NB{NB}_NL{NL}_a{alpha:.3f}".replace(".", "p")

    X, E, diag = anneal_optimize(
        N=N,
        w=w,
        restarts=int(args.restarts),
        iters=int(args.iters),
        seed=int(args.seed),
    )

    write_results(args.out, tag, types, alpha, X, E, diag)

    all_s = stats_angles(pairwise_angles(X))
    by = angle_lists_by_pairtype(X, types)
    verdict = classify_basic(N, types, X)
    bond_line = bond_angle_line(case_name, NB, NL, alpha, by)

    print("== done ==")
    print(f"tag: {tag}")
    print(f"alpha: {alpha:.6f}  NB={NB}  NL={NL}  N={N}")
    print(f"energy: {E:.6e}")
    print(f"angles(all): min={all_s['min_deg']:.3f}  p50={all_s['p50_deg']:.3f}  max={all_s['max_deg']:.3f}")
    print(verdict)
    print("BOND_ANGLE:", bond_line)  # <-- easy copy line
    print(f"saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
