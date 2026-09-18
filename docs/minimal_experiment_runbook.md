# 最小验证实验运行手册（Blackwell 设备：RTX 5060 8 GB / RTX 5090 32 GB）

对应研究计划 §16.1。目标：单宿主（Latent TransFuser）、单种子、四次微调，验证"接上保守负样本损失后交互场景更果断且不更危险"的因果链方向。**只看方向，不谈显著性**。

前提：设备驱动 ≥ 570（`nvidia-smi` 首行 CUDA Version ≥ 12.8）。本机（4060）已完成 cu128 环境构建与验证，见 `docs/env.md`"cu128 环境"节。

## 0. 目录约定
```
~/navsim_workspace/
├── navsim/            # v1.1，commit 3e8291b（Docker 镜像内已含）
├── exp/               # NAVSIM_EXP_ROOT：checkpoints/、metric_cache/、训练输出
├── dataset/           # OPENSCENE_DATA_ROOT：maps/、navsim_logs/{test,trainval}、sensor_blobs/{test,trainval}
└── env.sh             # 环境变量（含 CN_PROJECT_ROOT=项目目录）
~/E2E_planner/         # 本项目（git 分支 feat/objective-criteria-and-behavior-metrics）
```

## 1. 环境（约 30 分钟；Docker 约 5 分钟）
两种方式任选：
- **Docker**：`docker build -t negate:cu128 ~/E2E_planner`（或导入本机构建的镜像），`docker run --gpus all -it -v ~/navsim_workspace:/data -v ~/E2E_planner:/workspace/E2E_planner negate:cu128`。
- **conda**：`conda env create -f ~/E2E_planner/environment-cu128.yml`；然后 `pip install --no-deps "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"`；再 **`pip install --no-deps -e ~/navsim_workspace/navsim`**（必须带 `--no-deps`，否则 navsim 的 setup.py 会把 torch 钉回 2.0.1+cu117，Blackwell 上无法运行）；最后 `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"` 确认 2.7.1 与 sm_120。

验收：
```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"   # 含 sm_120
cd ~/E2E_planner && source ~/navsim_workspace/env.sh && export PYTHONPATH=$PWD CN_PROJECT_ROOT=$PWD
python tests/test_negaug_agent.py        # 需要第 3 节的 navtest 相机数据；PASS 且 max_neg_active_frac > 0
```

## 2. 元数据与地图（约 17 GB，10 分钟）
```bash
bash ~/navsim_workspace/dataset/download_metadata_only.sh
cd ~/navsim_workspace/dataset && mkdir -p navsim_logs && ln -sfn ../test_navsim_logs/test navsim_logs/test && ln -sfn ../trainval_navsim_logs/trainval navsim_logs/trainval
```

## 3. navtest 相机数据与指标缓存（评估用）
优先从本机复制（同一局域网约 30 分钟；否则重新下载约 2–5 小时）：
```bash
rsync -a --info=progress2 <本机>:~/navsim_workspace/dataset/sensor_blobs/test/  ~/navsim_workspace/dataset/sensor_blobs/test/     # 121 GB
rsync -a <本机>:~/navsim_workspace/exp/metric_cache/  ~/navsim_workspace/exp/metric_cache/                                         # 3 GB
rsync -a <本机>:~/navsim_workspace/exp/checkpoints/   ~/navsim_workspace/exp/checkpoints/                                          # 2 GB（官方 LTF 权重）
```
重新下载的替代：`bash ~/navsim_workspace/dataset/download_test_camera_only.sh` + `run_metric_caching.py train_test_split=navtest`。

验收：40 场景 dry-run 与本机结果一致
```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $CN_PROJECT_ROOT/configs train_test_split=navtest_dryrun \
  agent=ltf_camera_only agent.checkpoint_path=$NAVSIM_EXP_ROOT/checkpoints/ltf/ltf_seed_0.ckpt worker=single_machine_thread_pool experiment_name=dryrun
# 期望 Final average score ≈ 0.979（本机 cu117 值；cu128 差 ≤ 0.01）
```

## 4. navtrain 第 1 个当前帧压缩包（约 75 GB；网速 3–7 MB/s 下 3–7 小时）
```bash
bash ~/navsim_workspace/dataset/download_navtrain_chunk1.sh      # 流式解压，不落 tgz；完成后 sensor_blobs/trainval/ 约 300 个日志目录
tail -2 ~/navsim_workspace/dataset/download_navtrain_chunk1.log   # ALL DONE
```
磁盘：解压后约 75 GB；后续特征缓存约 6 GB。若磁盘 < 100 GB 可用，跳过第 3 节的 rsync，只保留 metric_cache 与权重，评估改在本机做（把 ckpt 传回本机）。

## 5. 微调子集（1 分钟）
```bash
cd ~/E2E_planner && python scripts/build_finetune_subset.py --sensor-dir $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
  --flags results/labels/scene_flags_navtrain.parquet --negatives results/labels/negatives_navtrain.parquet --bg-ratio 2 --name navtrain_chunk1_ft
```
验收：输出"交互约 2,400 / 背景约 4,800 / strict 负样本约 1,450"量级；"训练 N 帧 / 验证 M 帧"两者均非零（按官方日志划分）。生成 `configs/scene_filter/navtrain_chunk1_ft.yaml`、`configs/train_test_split/navtrain_chunk1_ft.yaml`、`results/navtrain_chunk1_ft_tokens.csv`。

## 6. 四次微调（每次约 30–40 分钟；5090 约 20 分钟）
| 序 | 命令 | 说明 |
|---|---|---|
| a | `FORCE_CACHE=true bash scripts/run_negaug_finetune.sh ft_lambda0 0.0` | 对照组；首次同时建特征缓存（约 30 分钟） |
| b | `bash scripts/run_negaug_finetune.sh ft_lambda0.1 0.1` | 处理组，λ=0.1（有效权重 1.0） |
| c | `bash scripts/run_negaug_finetune.sh ft_lambda0.3 0.3` | 处理组，λ=0.3（有效权重 3.0） |
| d | `bash scripts/run_negaug_finetune.sh ft_shuffled 0.1 0` | 错位负样本对照（缓存名 `negative_targets_shuffled`，首次多 10 分钟） |

显存档位：5060 8 GB 用默认 `BS=32 ACC=2`；5090 32 GB 可 `BS=64 ACC=1`，并可两任务并行。溢出时 `BS=16 ACC=4`。

每次检查点：日志 `Num training samples` 与第 5 节训练帧数一致；`train/loss` 下降；b/c/d 的 `train/neg_active_frac` 在 warm-up（前 20% 步）后 **≥ 1e-3**——若持续为 0，停止并按第 9 节回退。checkpoint 在 `$NAVSIM_EXP_ROOT/<run>/<时间戳>/lightning_logs/version_0/checkpoints/`，训练脚本会在 `$NAVSIM_EXP_ROOT/<run>/latest.ckpt` 建一个不含等号的链接——评估时**必须用这个链接**，Lightning 原文件名里的 `=` 会破坏 Hydra 覆盖语法。

## 7. 评估（每次约 15 分钟，共 1 小时）
```bash
bash scripts/eval_minimal_experiment.sh ft_lambda0 ft_lambda0.1 ft_lambda0.3 ft_shuffled     # 第一个为对照组
```
产出 `results/eval/minimal/`：每个 run 的交互子集 PDMS CSV、行为指标 parquet、`<run>_vs_ft_lambda0.csv` 配对差异表；终端打印分层配对差异（EP、PDMS、四个安全子项）与行为指标（不必要停车率、起步延迟 Δ、间隙接受率，括号为对照组）。

## 8. 判定（写入 `results/eval/minimal/DECISION.md`）
| 条 | 标准 | 通过 |
|---|---|---|
| 1 工程 | 四次训练与评估无报错；b/c/d 激活率 ≥ 1e-3；负样本命中率与子集一致 | 是 / 否 |
| 2 效果 | λ=0.1 相对 λ=0：间隙接受率 ↑、不必要停车率 ↓；NC / DAC / TTC 不 ↓ | |
| 3 剂量 | λ=0.3 的变化量 > λ=0.1，或开始损伤安全子项 | |
| 4 机制 | 错位对照的变化量明显小于 λ=0.1 正确配对 | |

四条全过 → 进入服务器阶段（研究计划 §16.2）；1 不过 → 修工程；2/3 不过 → 回到 Step 3/4 调整判据或损失；4 不过 → 效果来自正则化而非负监督，重新审视方法主张。最优一次补全量 navtest：`SPLIT=navtest bash scripts/eval_minimal_experiment.sh ft_lambda0 <best>`，确认不伤非交互场景（基线 83.5）。

## 9. 回退
- 激活率为 0：`bash scripts/run_negaug_finetune.sh ft_lambda0.1_infonce 0.1 null agent.loss_kind=infonce agent.temperature=1.0`；仍为 0 → 启用条件触发项（打分型宿主，见研究计划 §16.2）。
- 训练发散（loss NaN）：`agent.lr=1e-5`，或 `trainer.params.gradient_clip_val=1.0`。
- 评估 OOM / 线程：`worker.max_workers=2`。

## 10. 耗时汇总
| 阶段 | 5060 8 GB | 5090 32 GB |
|---|---|---|
| 环境 | 0.5 h | 0.5 h |
| 数据（元数据 + navtest 复制 + navtrain 1 包） | 4–8 h（网速） | 同 |
| 缓存 + 4 次训练 | 3 h | 1.5 h（可并行） |
| 评估 4 次 + 1 次全量 | 1.5 h | 1 h |
| **合计** | **约 9–13 h** | **约 7–11 h** |
