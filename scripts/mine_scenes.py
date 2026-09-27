#!/usr/bin/env python
"""Step 2：对 NAVSIM split 运行规则场景检测器（路径 C），输出每场景标志表与统计。

用法:
  source ~/navsim_workspace/env.sh
  python scripts/mine_scenes.py --split navtest --max-scenes 200 --out results/scene_flags_navtest_smoke.parquet
"""
import argparse, os, sys, time
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from conservative_negatives.scenes.rule_detector import detect, flags_to_dict

SPLIT_DIR = {"navtest": "test", "navtrain": "trainval", "navmini": "mini"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--turn-deg", type=float, default=45.0)
    a = ap.parse_args()

    devkit = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    yaml = devkit / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"
    cfg = OmegaConf.load(yaml)
    scene_filter = instantiate(cfg)
    if a.max_scenes:
        scene_filter.max_scenes = a.max_scenes
    loader = SceneLoader(
        data_path=data_root / "navsim_logs" / SPLIT_DIR[a.split],
        sensor_blobs_path=data_root / "sensor_blobs" / SPLIT_DIR[a.split],
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    print(f"{a.split}: {len(loader)} scenes loaded")

    rows, errors, t = [], 0, time.time()
    for i, tok in enumerate(loader.tokens):
        try:
            scene = loader.get_scene_from_token(tok)
            rows.append(flags_to_dict(detect(scene, turn_heading_deg=a.turn_deg)))
        except Exception as e:  # 记录但不中断
            errors += 1
            if errors <= 5:
                print(f"[warn] {tok}: {type(e).__name__}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(loader)}  {time.time()-t:.0f}s")
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    df.to_csv(a.out.with_suffix(".csv"), index=False)

    print(f"\nscenes={len(df)} errors={errors} time={time.time()-t:.0f}s")
    print("turn_type values:", df.connector_turn_type.value_counts(dropna=False).to_dict())
    print("connector_has_traffic_light:", df.connector_has_traffic_light.value_counts(dropna=False).to_dict())
    turns = df[df.connector_turn_type.isin(["LEFT", "RIGHT"])]
    print("turn connectors: n=%d, with traffic light=%d, unprotected(heading>=thr)=%d" % (
        len(turns), int((turns.connector_has_traffic_light == True).sum()), int(turns.unprotected_turn.sum())))
    for col in ["passes_lane_connector", "unprotected_turn", "lane_change", "reentry_from_stop", "merging", "unprotected_crossing", "expert_stopped_t0"]:
        print(f"  {col:24s} {int(df[col].sum()):6d}  ({100*df[col].mean():.1f}%)")
    print("heading_change_deg quantiles:", df.heading_change_deg.quantile([.5, .9, .99]).round(1).to_dict())
    print("start_delay_s (stopped at t0) quantiles:",
          df[df.expert_stopped_t0].expert_start_delay_s.quantile([.25, .5, .75]).round(2).to_dict())


if __name__ == "__main__":
    main()
