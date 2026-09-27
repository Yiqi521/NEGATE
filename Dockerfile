# NAVSIM v1.1 + 本项目，Blackwell 兼容环境（torch 2.7.1 + CUDA 12.8；同时支持 Ampere / Ada）。
# 构建：docker build -t negate:cu128 .        运行：docker run --gpus all -it -v <数据目录>:/data negate:cu128
# 数据与实验目录通过环境变量挂载（见 env.sh 约定）：
#   OPENSCENE_DATA_ROOT=/data/dataset  NUPLAN_MAPS_ROOT=/data/dataset/maps  NAVSIM_EXP_ROOT=/data/exp
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PATH=/opt/conda/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates build-essential rsync tmux libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
# 用 Miniforge（默认 conda-forge 频道）：Anaconda 默认频道在容器内需先接受服务条款，会使 conda create 失败
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p /opt/conda && rm /tmp/mf.sh && conda clean -afy

# 1) navsim 官方 devkit（锁定 v1.1 commit）
WORKDIR /workspace
RUN git clone -b v1.1 https://github.com/autonomousvision/navsim.git \
    && cd navsim && git checkout 3e8291bfa89ff247231e0227778840cd0a036896

# 2) 环境：python 3.9 + torch cu128 + navsim 依赖（去掉 torch 钉死行）+ nuplan-devkit(no-deps) + navsim
COPY requirements-cu128.txt /workspace/requirements-cu128.txt
RUN conda create -y -n navsim --override-channels -c conda-forge python=3.9 pip && conda clean -afy
SHELL ["conda", "run", "-n", "navsim", "/bin/bash", "-c"]
# 顺序很重要：navsim 的 setup.py 会把 torch 钉回 2.0.1，因此 navsim 用 --no-deps 安装，torch 最后装并校验版本
RUN pip install --no-cache-dir -r /workspace/requirements-cu128.txt \
    && pip install --no-cache-dir --no-deps "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2" \
    && pip install --no-cache-dir --no-deps -e /workspace/navsim \
    && pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
    && python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda"
# 注：构建期无 GPU，torch.cuda.get_arch_list() 为空；sm_120 支持在运行时用 `python -c "import torch; print(torch.cuda.get_arch_list())"` 确认

# 3) 本项目
COPY . /workspace/E2E_planner
ENV NAVSIM_DEVKIT_ROOT=/workspace/navsim CN_PROJECT_ROOT=/workspace/E2E_planner PYTHONPATH=/workspace/E2E_planner \
    NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=/data/dataset NUPLAN_MAPS_ROOT=/data/dataset/maps NAVSIM_EXP_ROOT=/data/exp
WORKDIR /workspace/E2E_planner
RUN python -c "import torch, navsim, nuplan; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "navsim", "/bin/bash", "-c"]
CMD ["bash"]
