#!/bin/bash
# 本机基线评估：官方 LTF 权重 × navtest（纯相机加载）。
# 用法: bash scripts/eval_ltf_navtest.sh <seed 0|1|2> [scene_filter=navtest]
set -e
SEED=${1:-0}
SPLIT=${2:-navtest}
source ~/anaconda3/etc/profile.d/conda.sh && conda activate navsim && source ~/navsim_workspace/env.sh
PROJ=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$PROJ:$PYTHONPATH
CKPT=$NAVSIM_EXP_ROOT/checkpoints/ltf/ltf_seed_${SEED}.ckpt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
[ -d "$NAVSIM_EXP_ROOT/metric_cache" ] || { echo "metric_cache missing"; exit 1; }
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  --config-dir $PROJ/configs \
  train_test_split=$SPLIT \
  agent=ltf_camera_only \
  agent.checkpoint_path=$CKPT \
  worker=single_machine_thread_pool \
  experiment_name=baseline_ltf_seed${SEED}_${SPLIT}
