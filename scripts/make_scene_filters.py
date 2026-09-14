#!/usr/bin/env python
"""由场景标志表生成 NAVSIM scene_filter yaml（unprotected_turn / merging / unprotected_crossing / interact）。

用法: python scripts/make_scene_filters.py --flags results/scene_flags_navtrain.parquet --prefix navtrain [--review-n 20]
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def write(path: Path, tokens, src):
    body = {"_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all",
            "num_history_frames": 4, "num_future_frames": 10, "frame_interval": 1, "has_route": True,
            "max_scenes": None, "log_names": None, "tokens": sorted(tokens)}
    with open(path, "w") as f:
        f.write(f"# 自动生成：scripts/make_scene_filters.py <- {src} ({len(tokens)} tokens)\n")
        yaml.safe_dump(body, f, sort_keys=False, default_flow_style=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("configs/scene_filter"))
    ap.add_argument("--review-n", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet(a.flags)
    if "unprotected_crossing" not in df:
        df["unprotected_crossing"] = False
    subsets = {
        "unprotected_turn": df[df.unprotected_turn],
        "merging": df[df.merging],
        "unprotected_crossing": df[df.unprotected_crossing & ~df.unprotected_turn],
        "interact": df[df.unprotected_turn | df.merging | df.unprotected_crossing],
    }
    a.outdir.mkdir(parents=True, exist_ok=True)
    print(f"{a.prefix}: total scenes {len(df)}, logs {df.log_name.nunique()}")
    for name, sub in subsets.items():
        write(a.outdir / f"{a.prefix}_{name}.yaml", sub.token.tolist(), a.flags.name)
        print(f"  {name:22s} {len(sub):6d} scenes  ({100*len(sub)/len(df):.1f}%)  logs {sub.log_name.nunique()}")
    if a.review_n:
        cols = ["token", "log_name", "subset", "heading_change_deg", "connector_turn_type", "ego_speed_t0", "expert_stopped_t0",
                "n_moving_vehicles_30m", "lane_change", "reentry_from_stop", "near_pudo_t0", "lateral_offset_t0"]
        parts = [subsets[k].sample(min(a.review_n, len(subsets[k])), random_state=i).assign(subset=k)
                 for i, k in enumerate(["unprotected_turn", "merging", "unprotected_crossing"])]
        out = Path("results") / f"review_sample_{a.prefix}.csv"
        pd.concat(parts)[cols].to_csv(out, index=False)
        print("review sample ->", out)


if __name__ == "__main__":
    main()
