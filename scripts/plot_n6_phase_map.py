import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(r"C:\UCM\chem")
csv_path = root / "paper_v1" / "data" / "n6_map_summary.csv"
out_path = root / "paper_v1" / "figures" / "n6_phase_map.png"

df = pd.read_csv(csv_path)

grp = (
    df.groupby("alpha", as_index=False)
      .agg({
          "BB_mean_deg": ["mean", "std"],
          "BL_p50_deg": ["mean", "std"],
          "LL_p50_deg": ["mean", "std"],
          "energy": ["mean", "std"],
      })
)

grp.columns = [
    "alpha",
    "BB_mean", "BB_std",
    "BL_mean", "BL_std",
    "LL_mean", "LL_std",
    "E_mean", "E_std",
]

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(grp["alpha"], grp["BB_mean"], marker="o", label="BB mean")
ax.plot(grp["alpha"], grp["BL_mean"], marker="s", label="BL median")
ax.plot(grp["alpha"], grp["LL_mean"], marker="^", label="LL median")

ax.fill_between(
    grp["alpha"],
    grp["BB_mean"] - grp["BB_std"],
    grp["BB_mean"] + grp["BB_std"],
    alpha=0.15
)
ax.fill_between(
    grp["alpha"],
    grp["BL_mean"] - grp["BL_std"],
    grp["BL_mean"] + grp["BL_std"],
    alpha=0.15
)
ax.fill_between(
    grp["alpha"],
    grp["LL_mean"] - grp["LL_std"],
    grp["LL_mean"] + grp["LL_std"],
    alpha=0.15
)

ax.axhline(90.0, linestyle="--", linewidth=1)

ax.set_xlabel("alpha")
ax.set_ylabel("angle (deg)")
ax.set_title("N=6, NB=2, NL=4: soft octahedral regime")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=220)
print(f"written: {out_path}")