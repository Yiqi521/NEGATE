#!/bin/bash
# 最小验证实验的评估：对每个微调 run 的最新 ckpt，在 navtest 交互子集上跑 PDMS + 行为指标，最后与 λ=0 对照做分层配对比较。
# 用法: bash scripts/eval_minimal_experiment.sh ft_lambda0 ft_lambda0.1 ft_lambda0.3 ft_shuffled   （第一个参数视为对照组）
set -e
source ~/anaconda3/etc/profile.d/conda.sh && conda activate navsim && source ~/navsim_workspace/env.sh
PROJ=$(cd "$(dirname "$0")/.." && pwd); export PYTHONPATH=$PROJ:$PYTHONPATH; export CN_PROJECT_ROOT=$PROJ
SPLIT=${SPLIT:-navtest_interact}; OUT=$PROJ/results/eval/minimal; mkdir -p $OUT
CONTROL=$1
for RUN in "$@"; do
  CKPT=$(ls -t $NAVSIM_EXP_ROOT/$RUN/*/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { echo "no checkpoint for $RUN"; exit 1; }
  echo "== $RUN  ($CKPT) =="
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $PROJ/configs \
    train_test_split=$SPLIT agent=ltf_negaug agent.checkpoint_path=$CKPT agent.host_checkpoint=null agent.lambda_neg=0 \
    worker=single_machine_thread_pool worker.max_workers=4 experiment_name=eval_${RUN}_${SPLIT} \
    > $OUT/${RUN}_pdm.log 2>&1 || echo "run_pdm_score exited non-zero for $RUN"
  grep -E "Number of successful|Final average" $OUT/${RUN}_pdm.log || true
  CSV=$(ls -t $NAVSIM_EXP_ROOT/eval_${RUN}_${SPLIT}/*/*.csv | head -1); cp "$CSV" $OUT/${RUN}_${SPLIT}.csv
  python $PROJ/eval/behavior_metrics.py --split navtest --negatives $PROJ/results/labels/negatives_navtest.parquet \
    --agent-config $PROJ/configs/agent/ltf_negaug.yaml --checkpoint $CKPT \
    --out $OUT/behavior_${RUN}.parquet --track --run-name behavior_${RUN} > $OUT/${RUN}_behavior.log 2>&1 \
    && grep -A 7 "行为指标" $OUT/${RUN}_behavior.log | head -8 || echo "behavior_metrics failed for $RUN"
done
echo; echo "######## 分层配对比较（相对对照组 $CONTROL）########"
for RUN in "${@:2}"; do
  echo; echo "---- $RUN vs $CONTROL ----"
  python $PROJ/eval/stratified_pdms.py --runs $OUT/${RUN}_${SPLIT}.csv --baseline $OUT/${CONTROL}_${SPLIT}.csv \
    --flags $PROJ/results/labels/scene_flags_navtest.parquet --negatives $PROJ/results/labels/negatives_navtest.parquet \
    --behavior $OUT/behavior_${RUN}.parquet --behavior-reference $OUT/behavior_${CONTROL}.parquet \
    --out $OUT/${RUN}_vs_${CONTROL}.csv | sed -n '/配对差异/,$p'
done
