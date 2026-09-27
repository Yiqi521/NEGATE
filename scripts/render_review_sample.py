#!/usr/bin/env python
"""为人工抽检样本渲染 BEV 图（当前帧地图 + 标注 + 专家未来 4 s 轨迹），输出 PNG 供二分类标注。

用法: python scripts/render_review_sample.py --sample results/review_sample_50_navtest.csv --split navtest --out results/review_png
"""
import argparse, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG

SPLIT_DIR = {"navtest": "test", "navtrain": "trainval"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    smp = pd.read_csv(a.sample)
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    sf.tokens = smp.token.tolist(); sf.log_names = smp.log_name.unique().tolist()
    loader = SceneLoader(root / "navsim_logs" / SPLIT_DIR[a.split], root / "sensor_blobs" / SPLIT_DIR[a.split],
                         sf, SensorConfig.build_no_sensors())
    cfg = TRAJECTORY_CONFIG.get("human", list(TRAJECTORY_CONFIG.values())[0])
    for _, r in smp.iterrows():
        if r.token not in loader.tokens:
            print("skip (not loaded):", r.token); continue
        scene = loader.get_scene_from_token(r.token)
        fig, ax = plot_bev_frame(scene, scene.scene_metadata.num_history_frames - 1)
        add_trajectory_to_bev_ax(ax, scene.get_future_trajectory(num_trajectory_frames=8), cfg)
        ax.set_title(f"{r.subset} | {r.token}\nheading={r.heading_change_deg:.0f}deg v0={r.ego_speed_t0:.1f}m/s "
                     f"moving_veh={r.n_moving_vehicles_30m} lane_change={r.lane_change} reentry={r.reentry_from_stop}", fontsize=8)
        fig.savefig(a.out / f"{r.subset}_{r.token}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    # 标注表模板
    smp.assign(label_is_target_scene="", label_note="").to_csv(a.out / "review_labels_template.csv", index=False)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
