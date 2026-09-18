#!/bin/bash
# 相机数据齐后一键执行：LTF 3 seeds × navtest 全量 → 分层统计（含人类专家对照）。
# 用法: bash scripts/eval_ltf_all.sh [seeds="0 1 2"]
set -e
SEEDS=${1:-"0 1 2"}
CONDA_ENV=${CONDA_ENV:-navsim}          # 可用 CONDA_ENV=navsim-cu128 切换环境
source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && source ~/navsim_workspace/env.sh
PROJ=$(cd "$(dirname "$0")/.." && pwd); export PYTHONPATH=$PROJ:$PYTHONPATH
OUT=${OUT:-$PROJ/results/eval}           # 输出目录（cu128 结果放 results/eval/cu128）
TAG=${TAG:-}                              # experiment_name 后缀，避免与旧结果目录混淆
[ -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test" ] || { echo "sensor_blobs/test 不存在（相机下载/整理未完成）"; exit 1; }
mkdir -p $OUT
RUNS=""
for s in $SEEDS; do
  echo "== LTF seed $s =="
  # 8 GB 显存：限制并发线程数（每线程各持一份模型），完整日志写入单独文件
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $PROJ/configs \
    train_test_split=navtest agent=ltf_camera_only agent.checkpoint_path=$NAVSIM_EXP_ROOT/checkpoints/ltf/ltf_seed_${s}.ckpt \
    worker=single_machine_thread_pool worker.max_workers=${MAX_WORKERS:-4} experiment_name=baseline_ltf_seed${s}_navtest${TAG} \
    > $OUT/ltf_seed${s}_run.log 2>&1 || echo "run_pdm_score exited non-zero for seed $s"
  grep -E "Traceback|Error|OutOfMemory" $OUT/ltf_seed${s}_run.log | head -5 || true
  grep -E "Number of successful|Number of failed|Final average" $OUT/ltf_seed${s}_run.log || true
  CSV=$(ls -t $NAVSIM_EXP_ROOT/baseline_ltf_seed${s}_navtest${TAG}/*/*.csv 2>/dev/null | head -1)
  [ -n "$CSV" ] || { echo "seed $s produced no CSV, abort"; exit 1; }
  cp "$CSV" $OUT/ltf_seed${s}_navtest.csv; RUNS="$RUNS $OUT/ltf_seed${s}_navtest.csv"
done
python $PROJ/eval/stratified_pdms.py --runs $RUNS --flags $PROJ/results/labels/scene_flags_navtest.parquet \
  --negatives $PROJ/results/labels/negatives_navtest.parquet --out $OUT/ltf_baseline_stratified.csv
