#!/bin/bash
# WSL2 / Ubuntu 22.04 一键环境（也适用于任何裸 Ubuntu 机器）：
#   Miniforge + conda 环境 "navsim"（torch 2.7.1 + CUDA 12.8，含 Blackwell sm_120）+ navsim v1.1 + ~/navsim_workspace/env.sh
# 用法（在 WSL 的 Ubuntu 终端里）:
#   bash /mnt/c/Users/<Windows 用户名>/PycharmProjects/NEGATE/scripts/setup/bootstrap_wsl.sh
# 可重复执行：已完成的步骤会跳过。安装顺序与仓库根目录 Dockerfile 一致（已在 4060 与容器内验证）。
# 目录布局与 docs/minimal_experiment_runbook.md §0 一致，scripts/*.sh 无需任何修改：
#   ~/navsim_workspace/{navsim,exp,dataset,env.sh}   ~/E2E_planner -> 本仓库   ~/anaconda3 -> ~/miniforge3
set -euo pipefail
PROJ=$(cd "$(dirname "$0")/../.." && pwd)
WS=${WS:-$HOME/navsim_workspace}
ENV_NAME=${ENV_NAME:-navsim}                 # scripts/*.sh 写死 conda activate navsim
NAVSIM_COMMIT=3e8291bfa89ff247231e0227778840cd0a036896
step() { echo; echo "==== $* ===="; }

step "0/6 前置检查"
grep -qi microsoft /proc/version && echo "WSL: yes" || echo "WSL: no（普通 Linux 也可继续）"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L; then
  echo "看不到 GPU。WSL 下请只更新 Windows 端 NVIDIA 驱动（≥ 570），不要在 Ubuntu 里装 Linux 驱动。"; exit 1
fi
if grep -q $'\r' "$PROJ/scripts/run_negaug_finetune.sh"; then
  echo "仓库脚本是 CRLF 行尾（Windows 检出，bash 会报 \$'\r'）。仓库已带 .gitattributes（eol=lf）；"
  echo "先提交或备份未提交的修改，再在仓库里执行: git rm -r --cached -q . && git reset -q --hard"; exit 1
fi

step "1/6 系统包（需要 sudo 密码）"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends git wget curl ca-certificates build-essential rsync tmux unzip pigz \
  libgl1 libglib2.0-0 libsm6 libxext6

step "2/6 Miniforge（~/miniforge3，并链接为 ~/anaconda3：仓库脚本 source ~/anaconda3/etc/profile.d/conda.sh）"
if [ ! -e "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  if [ ! -x "$HOME/miniforge3/bin/conda" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$HOME/miniforge3" && rm -f /tmp/mf.sh
  fi
  [ -e "$HOME/anaconda3" ] || ln -s "$HOME/miniforge3" "$HOME/anaconda3"
fi
# shellcheck disable=SC1091
source "$HOME/anaconda3/etc/profile.d/conda.sh"

step "3/6 工作区与 navsim v1.1（commit ${NAVSIM_COMMIT:0:7}）"
mkdir -p "$WS/exp" "$WS/dataset"
[ -d "$WS/navsim/.git" ] || git clone -q -b v1.1 https://github.com/autonomousvision/navsim.git "$WS/navsim"
git -C "$WS/navsim" checkout -q "$NAVSIM_COMMIT"
[ -e "$HOME/E2E_planner" ] || ln -s "$PROJ" "$HOME/E2E_planner"
git -C "$PROJ" config core.fileMode false     # /mnt/c 上的文件在 WSL 里全显示为可执行，避免 git status 满屏"模式变更"
for f in "$PROJ"/scripts/setup/download_*.sh; do ln -sfn "$f" "$WS/dataset/$(basename "$f")"; done   # 手册 §2–§4 的命令路径

step "4/6 conda 环境 $ENV_NAME（顺序固定：锁定依赖 → nuplan-devkit --no-deps → navsim --no-deps → 校验 torch）"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -y -q -n "$ENV_NAME" --override-channels -c conda-forge python=3.9 pip
conda activate "$ENV_NAME"
# requirements-cu128.lock.txt = 4060 上验证过的环境逐版本导出（含 torch 2.7.1+cu128）；不含 navsim / nuplan-devkit
# setuptools 65.5.1 是 requirements-cu128.txt 的钉死版本（pip freeze 不导出 setuptools，锁定文件里没有）
pip install -q -r "$PROJ/requirements-cu128.lock.txt" "setuptools==65.5.1" --extra-index-url https://download.pytorch.org/whl/cu128
pip install -q --no-deps "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"
pip install -q --no-deps -e "$WS/navsim"      # 必须 --no-deps：navsim 的 setup.py 会把 torch 钉回 2.0.1+cu117（sm_120 无内核）

step "5/6 env.sh"
cat > "$WS/env.sh" << ENVSH
# 由 E2E_planner/scripts/setup/bootstrap_wsl.sh 生成（$(date +%F)）
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$WS/dataset/maps"
export NAVSIM_EXP_ROOT="$WS/exp"
export NAVSIM_DEVKIT_ROOT="$WS/navsim"
export OPENSCENE_DATA_ROOT="$WS/dataset"
export CN_PROJECT_ROOT="$HOME/E2E_planner"
ENVSH
grep -q "navsim_workspace/env.sh" "$HOME/.bashrc" || echo "source $WS/env.sh" >> "$HOME/.bashrc"
# shellcheck disable=SC1091
source "$WS/env.sh"

step "6/6 验收（docs/minimal_experiment_runbook.md §1）"
python - << 'PY'
import torch, navsim, nuplan, hydra, pytorch_lightning as pl
assert torch.__version__.startswith("2.7.1"), f"torch 版本不对: {torch.__version__}（navsim 依赖把 torch 改回去了？）"
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "torch 看不到 GPU"
arch = torch.cuda.get_arch_list(); assert "sm_120" in arch, arch
x = torch.randn(1024, 1024, device="cuda"); float((x @ x).sum())
print(f"OK torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | sm_120 in arch | lightning {pl.__version__}")
PY
python -c "import sys; sys.path.insert(0, '$PROJ'); import conservative_negatives.train.negaug_agent as m; print('OK adapter import:', m.NegativeAugmentedAgent.__name__)"
echo
echo "环境就绪。下一步（数据）："
echo "  bash $WS/dataset/download_metadata_only.sh            # 元数据 + 地图，约 8.5 GB 下载 / 17 GB 解压"
echo "  然后在 tmux 里并行：download_test_camera_only.sh、download_navtrain_chunk1.sh、download_ltf_checkpoints.sh 与 metric cache"
echo "  详见 $PROJ/scripts/setup/README.md"
