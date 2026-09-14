#!/usr/bin/env python
"""Step 0' 预检（路径 A）：检查 OpenScene/NAVSIM 日志 pickle 是否携带场景类型字段。

用法:
  python scripts/precheck_scene_tags.py $OPENSCENE_DATA_ROOT/test_navsim_logs/test [--max-logs 50]
输出:
  - 帧字典的键集合
  - 名称中含 scenario / tag / type 的候选键及其取值分布
  - 日志数、帧数
"""
import argparse, collections, pickle, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--max-logs", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None, help="可选：把候选键取值分布写成 csv")
    a = ap.parse_args()

    log_files = sorted(p for p in a.log_dir.rglob("*.pkl"))
    if not log_files:
        sys.exit(f"no .pkl under {a.log_dir}")
    print(f"found {len(log_files)} logs, inspecting first {min(a.max_logs, len(log_files))}")

    keys = collections.Counter()
    cand_vals = collections.defaultdict(collections.Counter)
    n_frames = 0
    for p in log_files[: a.max_logs]:
        frames = pickle.load(open(p, "rb"))
        for f in frames:
            n_frames += 1
            keys.update(f.keys())
            for k, v in f.items():
                kl = k.lower()
                if any(s in kl for s in ("scenario", "tag", "type")) and kl not in ("gt_names",):
                    if isinstance(v, (list, tuple, set)):
                        for x in v:
                            cand_vals[k][str(x)] += 1
                    else:
                        cand_vals[k][str(v)] += 1

    print(f"\nframes inspected: {n_frames}")
    print("\n== frame keys (count of frames having key) ==")
    for k, c in keys.most_common():
        print(f"  {k:32s} {c}")

    print("\n== candidate scenario-type keys ==")
    if not cand_vals:
        print("  NONE -> 路径 A 不可行，转路径 B（nuPlan DB 回连）或路径 C（规则检测器）")
    for k, cnt in cand_vals.items():
        print(f"  [{k}] {len(cnt)} distinct values; top 40:")
        for v, c in cnt.most_common(40):
            print(f"     {v:55s} {c}")
        hits = [v for v in cnt if "unprotected" in v or "changing_lane" in v or "pickup_dropoff" in v]
        print(f"  -> 与本研究相关的取值: {hits}")

    if a.out and cand_vals:
        import csv
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["key", "value", "count"])
            for k, cnt in cand_vals.items():
                for v, c in cnt.most_common():
                    w.writerow([k, v, c])
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
