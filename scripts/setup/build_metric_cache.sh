#!/bin/bash
# navtest 指标缓存（PDM 评分用；只需元数据 + 地图，不需要传感器数据，可与相机下载并行）。
# 旧机 22 线程约 25 分钟、3.1 GB；本机 16 线程约 40 分钟。WSL 内存吃紧时用 THREADS=8。
set -euo pipefail
# shellcheck disable=SC1091
source ~/anaconda3/etc/profile.d/conda.sh && conda activate navsim && source ~/navsim_workspace/env.sh
THREADS=${THREADS:-null}
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  worker.threads_per_node=$THREADS
n=$(find "$NAVSIM_EXP_ROOT/metric_cache" -name metric_cache.pkl | wc -l)
echo "metric_cache.pkl: $n 个（旧机 12,146）"
