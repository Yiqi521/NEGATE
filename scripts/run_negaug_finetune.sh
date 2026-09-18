#!/bin/bash
# 最小验证实验：在本机 8 GB GPU 上微调 LTF + 保守负样本损失。
# 用法: bash scripts/run_negaug_finetune.sh <run_name> <lambda> [shuffle_seed|null] [extra hydra overrides...]
#   bash scripts/run_negaug_finetune.sh ft_lambda0    0.0
#   bash scripts/run_negaug_finetune.sh ft_lambda0.1  0.1
#   bash scripts/run_negaug_finetune.sh ft_lambda0.3  0.3
#   bash scripts/run_negaug_finetune.sh ft_shuffled   0.1 0
set -e
RUN=$1; LAMBDA=$2; SHUFFLE=${3:-null}; shift 3 2>/dev/null || shift $#
source ~/anaconda3/etc/profile.d/conda.sh && conda activate navsim && source ~/navsim_workspace/env.sh
PROJ=$(cd "$(dirname "$0")/.." && pwd); export PYTHONPATH=$PROJ:$PYTHONPATH; export CN_PROJECT_ROOT=$PROJ
SPLIT=${SPLIT:-navtrain_chunk1_ft}
EPOCHS=${EPOCHS:-10}; BS=${BS:-32}; ACC=${ACC:-2}
# total_steps = 训练样本数 / (BS*ACC) * EPOCHS，用子集 token 表中训练侧数量估算
N_TRAIN=$(python - << PY
import pandas as pd, yaml, os
t = pd.read_csv("$PROJ/results/${SPLIT}_tokens.csv")
split = yaml.safe_load(open(os.environ["NAVSIM_DEVKIT_ROOT"] + "/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"))
print(int(t.log_name.isin(set(split["train_logs"])).sum()))
PY
)
STEPS=$(( N_TRAIN / (BS*ACC) * EPOCHS )); [ $STEPS -lt 1 ] && STEPS=1
echo "run=$RUN lambda=$LAMBDA shuffle=$SHUFFLE split=$SPLIT n_train=$N_TRAIN total_steps=$STEPS"
mkdir -p $PROJ/results/train
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py --config-dir $PROJ/configs \
  agent=ltf_negaug agent.lambda_neg=$LAMBDA agent.shuffle_seed=$SHUFFLE agent.total_steps=$STEPS \
  train_test_split=$SPLIT experiment_name=$RUN \
  cache_path=$NAVSIM_EXP_ROOT/negaug_cache_${SPLIT} force_cache_computation=${FORCE_CACHE:-false} \
  dataloader.params.batch_size=$BS dataloader.params.num_workers=6 \
  trainer.params.max_epochs=$EPOCHS trainer.params.accumulate_grad_batches=$ACC \
  trainer.params.strategy=auto +trainer.params.devices=1 trainer.params.precision=16-mixed \
  "$@" 2>&1 | tee $PROJ/results/train/${RUN}.log | grep -E "Epoch|neg_active|loss|Error|Traceback|Num training|Num validation|NegAug" | grep -v "it/s"
# Lightning 的 ckpt 文件名含 "="（epoch=1-step=2.ckpt），会破坏 Hydra 覆盖语法；建一个不含 "=" 的 latest.ckpt 链接供评估脚本使用
CK=$(ls -t $NAVSIM_EXP_ROOT/$RUN/*/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then ln -sfn "$CK" $NAVSIM_EXP_ROOT/$RUN/latest.ckpt; echo "checkpoint: $CK"; echo "symlink:    $NAVSIM_EXP_ROOT/$RUN/latest.ckpt"; else echo "no checkpoint produced"; fi
