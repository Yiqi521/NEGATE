#!/usr/bin/env python
"""最小验证实验的微调子集：以本机已下载的 navtrain 传感器日志为范围，取其中全部交互场景 + 若干倍随机背景帧。

用法:
  python scripts/build_finetune_subset.py --sensor-dir $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
      --flags results/labels/scene_flags_navtrain.parquet --negatives results/labels/negatives_navtrain.parquet \
      --bg-ratio 2 --name navtrain_chunk1_ft --out configs
输出:
  configs/scene_filter/<name>.yaml、configs/train_test_split/<name>.yaml、results/<name>_tokens.csv、统计
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd, yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-dir", type=Path, required=True, help="已解压的 navtrain 传感器目录（子目录 = 日志名）")
    ap.add_argument("--flags", type=Path, required=True); ap.add_argument("--negatives", type=Path, required=True)
    ap.add_argument("--bg-ratio", type=float, default=2.0, help="背景帧 = bg_ratio × 交互场景数")
    ap.add_argument("--name", default="navtrain_chunk1_ft"); ap.add_argument("--out", type=Path, default=Path("configs"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    logs_local = sorted(d.name for d in a.sensor_dir.iterdir() if d.is_dir())
    flags = pd.read_parquet(a.flags)
    neg = pd.read_parquet(a.negatives)
    strict_tokens = set(neg[neg.is_negative_strict].token)
    f = flags[flags.log_name.isin(logs_local)].copy()
    f["interact"] = f.unprotected_turn | f.merging | f.unprotected_crossing
    inter = f[f.interact]
    bg_pool = f[~f.interact]
    n_bg = min(len(bg_pool), int(a.bg_ratio * len(inter)))
    bg = bg_pool.sample(n_bg, random_state=a.seed)
    sub = pd.concat([inter, bg])
    sub["has_strict_negative"] = sub.token.isin(strict_tokens)

    print(f"本地日志 {len(logs_local)} 个（navtrain 共 {flags.log_name.nunique()}）")
    print(f"其中场景 {len(f)}；交互 {len(inter)}（有 strict 负样本 {int(inter.token.isin(strict_tokens).sum())}）；背景抽 {n_bg}")
    print(f"子集合计 {len(sub)} 帧，覆盖日志 {sub.log_name.nunique()} 个")
    # NAVSIM 训练按 train_logs / val_logs 拆分；估算落入各方的数量
    dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    split = yaml.safe_load(open(dev / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"))
    tr, va = set(split["train_logs"]), set(split["val_logs"])
    print(f"按官方日志划分：训练 {int(sub.log_name.isin(tr).sum())} 帧 / 验证 {int(sub.log_name.isin(va).sum())} 帧")

    body = {"_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all", "num_history_frames": 4, "num_future_frames": 10,
            "frame_interval": 1, "has_route": True, "max_scenes": None, "log_names": sorted(sub.log_name.unique().tolist()),
            "tokens": sorted(sub.token.tolist())}
    (a.out / "scene_filter").mkdir(parents=True, exist_ok=True); (a.out / "train_test_split").mkdir(parents=True, exist_ok=True)
    with open(a.out / "scene_filter" / f"{a.name}.yaml", "w") as fh:
        fh.write(f"# 微调子集：{len(inter)} 交互 + {n_bg} 背景 = {len(sub)} 帧，来自本地 {len(logs_local)} 个日志\n")
        yaml.safe_dump(body, fh, sort_keys=False, default_flow_style=False)
    (a.out / "train_test_split" / f"{a.name}.yaml").write_text(f"defaults:\n  - scene_filter: {a.name}\n\ndata_split: trainval\n")
    sub[["token", "log_name", "interact", "has_strict_negative", "unprotected_turn", "merging", "unprotected_crossing"]].to_csv(
        Path("results") / f"{a.name}_tokens.csv", index=False)
    print("->", a.out / "scene_filter" / f"{a.name}.yaml")


if __name__ == "__main__":
    main()
