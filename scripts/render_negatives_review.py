#!/usr/bin/env python
"""为负样本人工审查渲染 BEV：专家（绿）与保守负样本（红）同图，标题给出算子、EP 比与 C3 字段。

用法: python scripts/render_negatives_review.py --negatives results/negatives_navtest_v4.parquet --per-class 20 --out results/review_negatives_png
"""
import argparse, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from conservative_negatives.generators.operators import (g1_delayed_departure, g2_speed_scale, g3_early_brake, g3b_brake_to_stop, g4_large_gap)

OPS = {"G1": g1_delayed_departure, "G2": g2_speed_scale, "G3": g3_early_brake, "G3b": g3b_brake_to_stop}


def rebuild(op, params, exp, conflict_clear_s):
    p = eval(params)
    if op == "G4":
        return g4_large_gap(exp, None if p.get("conflict_clear_s", -1) < 0 else p["conflict_clear_s"], p["extra_wait_s"])
    return OPS[op](exp, **p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(a.negatives); d = d[d.is_negative].copy()
    d["cls"] = np.where(d.unprotected_turn, "turn", np.where(d.merging, "merging", "crossing"))
    smp = pd.concat([g.sample(min(a.per_class, len(g)), random_state=7) for _, g in d.groupby("cls")])

    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    sf.tokens = smp.token.unique().tolist(); sf.log_names = smp.log_name.unique().tolist()
    loader = SceneLoader(root / "navsim_logs" / ("test" if a.split == "navtest" else "trainval"),
                         root / "sensor_blobs" / ("test" if a.split == "navtest" else "trainval"), sf, SensorConfig.build_no_sensors())
    cfg_h = TRAJECTORY_CONFIG.get("human", list(TRAJECTORY_CONFIG.values())[0])
    # 负样本与专家共路径：用更大的空心红圆 + 虚线，zorder 更高，保证重叠时可辨
    cfg_n = dict(cfg_h); cfg_n.update({"line_color": "#d62728", "marker_edge_color": "#d62728", "fill_color": "none",
                                        "line_color_alpha": 0.9, "line_style": "--", "marker": "o", "marker_size": 11,
                                        "line_width": 1.2, "zorder": 5})
    rows = []
    for i, r in smp.reset_index(drop=True).iterrows():
        if r.token not in loader.tokens:
            continue
        scene = loader.get_scene_from_token(r.token)
        exp = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(float)
        neg = rebuild(r.operator, r.params, exp, r.conflict_clear_s)
        fig, ax = plot_bev_frame(scene, scene.scene_metadata.num_history_frames - 1)
        add_trajectory_to_bev_ax(ax, Trajectory(exp.astype(np.float32)), cfg_h)
        try:
            add_trajectory_to_bev_ax(ax, Trajectory(neg.astype(np.float32)), cfg_n)
        except Exception:
            ax.plot(neg[:, 1] * -1, neg[:, 0], "r.-")  # 兜底
        from conservative_negatives.generators.operators import _speed_profile
        ins = ax.inset_axes([0.62, 0.02, 0.36, 0.22]); t = np.arange(1, 9) * 0.5
        ins.plot(t, _speed_profile(exp), color="#2ca02c", lw=1.5, label="expert"); ins.plot(t, _speed_profile(neg), color="#d62728", lw=1.5, ls="--", label="negative")
        ins.set_xlabel("t [s]", fontsize=6); ins.set_ylabel("v [m/s]", fontsize=6); ins.tick_params(labelsize=6); ins.legend(fontsize=6, loc="upper left"); ins.patch.set_alpha(0.85)
        gap = "None" if pd.isna(r.ref_min_gap_s) else f"{r.ref_min_gap_s:.1f}s"
        ax.set_title(f"[{r.cls}] {r.token} | {r.operator} {r.params}\nEP ratio={r.ep_ratio:.2f}  expert EP={r.exp_ep:.2f}  ref_gap={gap}  v0={r.ego_speed_t0:.1f}m/s", fontsize=8)
        fn = f"{r.cls}_{i:03d}_{r.token}_{r.operator}.png"
        fig.savefig(a.out / fn, dpi=110, bbox_inches="tight"); plt.close(fig)
        rows.append(dict(file=fn, token=r.token, cls=r.cls, operator=r.operator, params=r.params, ep_ratio=round(r.ep_ratio, 2),
                         ref_min_gap_s=r.ref_min_gap_s, label_is_unjustified_conservative="", label_note=""))
    pd.DataFrame(rows).to_csv(a.out / "review_negatives_template.csv", index=False)
    print("rendered", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
