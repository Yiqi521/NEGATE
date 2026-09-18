# 环境记录（Step 0）

生成时间：2026-09-14 17:06 JST

## 硬件（当前开发机）
| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM, driver 580.173.02 |
| CPU / 内存 | 22 线程 / 62 GB |
| 磁盘 | 单盘 935 GB，剩余约 159 GB（2026-09-14 17:00）；用户清理后剩余约 299 GB（17:20） |

**结论**：本机只能承担 Step 1–3（定义、场景挖掘、负样本生成与过滤）与小规模调试；navtrain 传感器数据（445 GB）与 TransFuser 训练（官方：1 GPU·天，默认 batch 64）需迁移到实验室服务器（≥ 24 GB 显存，≥ 1.2 TB 磁盘）。

## 代码版本
| 组件 | 路径 | 版本 |
|---|---|---|
| navsim | ~/navsim_workspace/navsim | 分支 v1.1，commit 3e8291bfa89ff247231e0227778840cd0a036896（2025-06-05） |
| nuplan-devkit | 经 navsim requirements 安装 | nuplan-devkit-v1.2（git tag） |
| torch | conda env navsim | 2.0.1 / torchvision 0.15.2 / pytorch-lightning 2.2.1 |
| python | conda env navsim | 3.9 |

## 环境变量
见 ~/navsim_workspace/env.sh（已由 ~/.bashrc 自动 source）。

## 数据（仅元数据与地图，本机）
| 数据 | 路径 | 状态 |
|---|---|---|
| maps | $OPENSCENE_DATA_ROOT/maps | 下载中 |
| navtest 元数据 | $OPENSCENE_DATA_ROOT/test_navsim_logs | 下载中 |
| navtrain 元数据 | $OPENSCENE_DATA_ROOT/trainval_navsim_logs | 下载中 |
| navtest 相机 blob（约 120 GB） | $OPENSCENE_DATA_ROOT/sensor_blobs/test | 下载中（脚本 download_test_camera_only.sh，剩余 < 60 GB 自动停止） |
| navtest 激光雷达 blob（约 87 GB） | — | 暂缓（TransFuser 需要；LTF 不需要） |
| navtrain 传感器 blob（445 GB） | — | 本机不下载，服务器执行 |

## Step 0' 预检记录
- 场景标签路径 A 不可行（详见 decision_scene_tags.md），已切换路径 C 规则检测器：`conservative_negatives/scenes/rule_detector.py`，运行脚本 `scripts/mine_scenes.py`。
- 200 场景冒烟测试（navtest）：unprotected_turn 4.0%，merging 7.0%，expert_stopped_t0 8.5%；nuPlan 的 `LaneConnector.turn_type()` 未实现，转向类型改由基线路径首尾航向差推断。
- navtest 指标缓存：`$NAVSIM_EXP_ROOT/metric_cache`（运行中）。

## 2026-09-18 决定
- **navtrain 传感器数据不在本机下载**；最小验证实验在另一台 Blackwell（5060 / 5090）设备上执行，本机只负责：cu128 环境预构建与验证、打包、运行手册、对照标签。
- 已启动的 `navtrain_current_1.tgz` 流式下载于 15:53 启动、约 16:20 按用户要求终止，残留部分解压已删除。

## cu128 环境（Blackwell 兼容；2026-09-18 起）
目的：5060 / 5090（sm_120）无法运行锁定的 torch 2.0.1+cu117（编译内核只到 sm_86）。在本机 4060（sm_89，cu128 轮子同样含其内核）预构建并验证，再打包到设备。

| 项目 | cu117（现有 `navsim`） | cu128（`navsim-cu128`） |
|---|---|---|
| torch / CUDA | 2.0.1+cu117 / 11.7 | 2.7.1+cu128 / 12.8（cuDNN 9.7.1，NCCL 2.26.2） |
| 编译架构 | sm_37…sm_86 | sm_75, sm_80, sm_86, sm_90, sm_100, **sm_120**, compute_120（Ada 以 sm_86 二进制运行，本机 4060 实测正常） |
| 适配器测试 `tests/test_negaug_agent.py` | PASS：hit_rate 0.667，max_neg_active_frac 0.167，最后批 loss_host 26.09–26.83 | PASS：hit_rate 0.667，max_neg_active_frac 0.167，loss_host 26.78（一致） |
| LTF 40 场景 dry-run PDMS | 0.9794 | 0.9794；逐 token PDMS 与 EP 最大差 0.0000 |
| 显存峰值 batch 4/8/16/32（前向+反向+Adam） | 1.38 / 2.33 / 3.67 / 6.36 GB | 1.52 / 2.22 / 3.65 / 6.41 GB |
| 步时 batch 32 | 0.99 s（含首步预热） | 0.51 s（预热后 3 步均值；两次测量口径不同，仅说明 cu128 不更慢） |
| 训练冒烟（navtest 有负样本的 48 场景，2 epoch，batch 8 × 累积 4） | — | 通过：40 训练 / 8 验证样本；loss_neg 0.35，neg_active_frac 0.43–0.45，每批有效负样本 23.4，λ_eff 0.8 → 1.0；ckpt 与 latest.ckpt 链接生成 |
| LTF 三种子 navtest 全量 PDMS | 84.1 / 83.1 / 83.3（均 83.5 ± 0.45） | 待填 |
| 打包 | — | `environment-cu128.yml`（含 cu128 额外索引；navsim / nuplan-devkit 需按文件头注释另装）、`requirements-cu128.txt` + `.lock.txt`、`Dockerfile` → 镜像 `negate:cu128`（22.2 GB；基础镜像 nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 + Miniforge）。容器内 GPU 测试通过：torch 2.7.1+cu128，架构含 sm_120；容器内 LTF 40 场景 dry-run PDMS 0.9794 与主机一致 |

安装步骤（已脚本化于 Dockerfile；**顺序很重要**）：`conda create -n navsim-cu128 python=3.9` → `pip install -r requirements-cu128.txt`（navsim 依赖去掉 torch/torchvision 钉死行）→ `pip install --no-deps nuplan-devkit@v1.2` → **`pip install --no-deps -e navsim`** → **最后** `pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128` → 断言 `torch.__version__` 以 2.7.1 开头且 `sm_120` 在架构列表中。

**踩坑记录（2026-09-18）**：首次构建时先装 torch cu128、最后 `pip install -e navsim`，结果 navsim 的 `setup.py` 直接读取 `requirements.txt` 作为 `install_requires`，把 torch 卸载并钉回 2.0.1+cu117——环境看起来建成了，但 `torch.cuda.get_arch_list()` 仍无 sm_120。此时做的 40 场景一致性对比（0 差异）无效，已重做。

**冒烟中发现并修复的三处工程问题**：(1) 官方训练配置无 `devices` 键，覆盖需写 `+trainer.params.devices=1`；(2) Lightning 的 ckpt 文件名含 `=`（`epoch=1-step=4.ckpt`）会破坏 Hydra 覆盖语法，训练脚本现在生成 `<run>/latest.ckpt` 链接供评估；(3) 标签文件必须与训练 split 一致（navtest 场景用 `negatives_navtest.parquet`），否则负样本命中为零而训练照常进行——适配器现在在 50 步内零命中时打印告警。

**Docker 构建踩坑**：(1) Miniconda 默认频道在容器内要求先接受 Anaconda 服务条款，`conda create` 直接失败 → 改用 Miniforge 并 `--override-channels -c conda-forge`；(2) 构建期没有 GPU，`torch.cuda.get_arch_list()` 为空，不能在 Dockerfile 里断言 sm_120，只断言 torch 版本与 `torch.version.cuda == "12.8"`，架构检查放到运行时。
镜像迁移：目标设备有网时直接 `docker build`（约 25 分钟）；无网时 `docker save negate:cu128 | gzip > negate_cu128.tar.gz`（约 10 GB）拷贝后 `docker load`。
(3) OpenCV 需要系统库 `libgl1 libglib2.0-0 libsm6 libxext6`，基础镜像缺失会在导入时报 `libGL.so.1`；(4) metric cache 的索引 CSV 是绝对路径，容器挂载点不同或迁移到别的机器都要先跑 `scripts/relocate_metric_cache.py`。
