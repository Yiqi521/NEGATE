#!/usr/bin/env python
"""NAVSIM metric cache 迁移：索引 CSV（metadata/metric_cache_metadata_node_*.csv）记录的是生成机器上的绝对路径，
复制到新机器 / 容器后需要把路径前缀改写为当前位置，否则 run_pdm_score 报 FileNotFoundError。

用法: python scripts/relocate_metric_cache.py <metric_cache 目录>          # 自动把前缀改为该目录的绝对路径
      python scripts/relocate_metric_cache.py <metric_cache 目录> --dry-run
"""
import argparse, csv, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cache_dir", type=Path); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = a.cache_dir.resolve(); meta = root / "metadata"
    files = sorted(meta.glob("metric_cache_metadata_node_*.csv"))
    if not files:
        sys.exit(f"no metadata csv under {meta}")
    for f in files:
        rows = list(csv.reader(open(f)))
        header, body = rows[0], rows[1:]
        fixed, missing = [], 0
        for r in body:
            old = Path(r[0])
            # 路径形如 <old_root>/<log>/<scenario_type>/<token>/metric_cache.pkl → 取最后 4 段接到新 root
            new = root.joinpath(*old.parts[-4:])
            if not new.exists():
                missing += 1
            fixed.append([str(new)] + r[1:])
        print(f"{f.name}: {len(body)} entries, prefix -> {root}, missing files: {missing}")
        if not a.dry_run:
            with open(f, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(header); w.writerows(fixed)
    print("done" if not a.dry_run else "dry-run only")


if __name__ == "__main__":
    main()
