#!/bin/bash
# 官方 Latent TransFuser 权重 3 种子（每个 0.67 GB），来源 huggingface.co/autonomousvision/navsim_baselines。
# 放到 $NAVSIM_EXP_ROOT/checkpoints/ltf/ltf_seed_{0,1,2}.ckpt（configs/agent/ltf_negaug.yaml 与 scripts/eval_ltf_navtest.sh 的约定路径）。
source "$(dirname "$(readlink -f "$0")")/_common.sh"
D="$NAVSIM_EXP_ROOT/checkpoints/ltf"; mkdir -p "$D"
for s in 0 1 2; do
  wget -c --tries=30 --progress=dot:mega -P "$D" "https://huggingface.co/autonomousvision/navsim_baselines/resolve/main/ltf/ltf_seed_${s}.ckpt"
done
ls -la "$D"
log "LTF CHECKPOINTS DONE"
