# 新设备上线：Windows 11 + WSL2（RTX 5060 Ti 8 GB）

目标：在本机跑通 `docs/minimal_experiment_runbook.md`（研究计划 §16.1 最小验证实验）。
本目录的脚本把旧机上只存在于 `~/navsim_workspace/` 的环境脚本与下载脚本收进了仓库，任何裸 Ubuntu 机器也可直接用（跳过第 1 步）。

| 脚本 | 作用 | 下载 / 落盘 | 耗时（参考） |
|---|---|---|---|
| `bootstrap_wsl.sh` | Miniforge + conda 环境 `navsim`（torch 2.7.1+cu128，含 sm_120）+ navsim v1.1 + `env.sh` + 验收 | 约 15 GB | 30–40 min |
| `download_metadata_only.sh` | 地图 + navtest / navtrain 元数据 + `navsim_logs/{test,trainval}` 链接（手册 §2） | 8.5 GB / 17 GB | 10–30 min |
| `download_test_camera_only.sh` | navtest 相机 32 包，不含激光雷达（手册 §3） | 125 GB / 121 GB | 网速决定 |
| `download_navtrain_chunk1.sh` | navtrain 第 1 个当前帧包，含官方 MD5 校验（手册 §4） | 54.5 GB / 约 75 GB | 网速决定 |
| `download_ltf_checkpoints.sh` | 官方 LTF 权重 3 种子 → `$NAVSIM_EXP_ROOT/checkpoints/ltf/` | 2 GB | 几分钟 |
| `build_metric_cache.sh` | navtest 指标缓存（只需元数据 + 地图） | 3 GB | 约 40 min |

所有下载脚本都可重复执行：压缩包断点续传，每包解压完成后写标记，重跑跳过；任一步开始前若磁盘（取 WSL 虚拟盘与宿主 C: 盘的较小值）会低于 `MIN_FREE_GB`（默认 40）则自动停止。2026-09-27 已核对全部下载地址可达、大小与上表一致。

## 1. 安装 WSL2（管理员 PowerShell，需重启一次）
```powershell
wsl --install -d Ubuntu-22.04
```
重启后按提示建 Linux 用户。`C:\Users\su090\.wslconfig` 已写好（内存上限 24 GB、swap 16 GB；默认只给一半即 16 GB）。
进 Ubuntu 后先确认 GPU：`nvidia-smi` 能看到 RTX 5060 Ti。**不要**在 Ubuntu 里装 Linux 版 NVIDIA 驱动或 CUDA Toolkit，Windows 驱动（≥ 570，本机 596.49）已提供 WSL 支持，torch 轮子自带 CUDA 12.8 运行时。

## 2. 环境（Ubuntu 终端）
```bash
bash /mnt/c/Users/su090/PycharmProjects/NEGATE/scripts/setup/bootstrap_wsl.sh
```
会要一次 sudo 密码（装 `build-essential`、`libgl1` 等系统包）。结束时打印 `OK torch 2.7.1+cu128 | CUDA 12.8 | NVIDIA GeForce RTX 5060 Ti | sm_120 in arch` 即成功。
布局与手册 §0 一致：`~/navsim_workspace/{navsim,exp,dataset,env.sh}`，`~/E2E_planner` 链接到本仓库，`~/anaconda3` 链接到 `~/miniforge3`，因此 `scripts/*.sh` 不用改。

**代码在 /mnt/c，数据必须在 WSL 自己的 ext4（`~/navsim_workspace`）**：/mnt/c 的读写经 9P 协议，比 ext4 慢一个数量级，放数据会拖垮缓存与训练。

## 3. 数据（建议在 tmux 里并行）
```bash
source ~/.bashrc
bash ~/navsim_workspace/dataset/download_metadata_only.sh          # 先做这一步，其余依赖它

tmux new -s dl
export DL_TMP=/mnt/c/Users/su090/Downloads/navsim_dl                # 压缩包放 Windows 盘：删后空间立即归还（WSL 虚拟盘不会自动缩小）
bash ~/navsim_workspace/dataset/download_navtrain_chunk1.sh         # 窗口 1
bash ~/navsim_workspace/dataset/download_test_camera_only.sh        # 窗口 2（Ctrl-b c 新开）
bash ~/navsim_workspace/dataset/download_ltf_checkpoints.sh && bash ~/E2E_planner/scripts/setup/build_metric_cache.sh   # 窗口 3
```
`tail -2 ~/navsim_workspace/dataset/download_navtrain_chunk1.log` 末行 `ALL DONE` 即 navtrain 完成。
**旧机可达时更快**：按手册 §3 用 rsync 复制 `sensor_blobs/test`、`exp/metric_cache`、`exp/checkpoints`，然后 `python ~/E2E_planner/scripts/relocate_metric_cache.py ~/navsim_workspace/exp/metric_cache`；navtrain 仍需下载。

磁盘预算（C: 盘剩余 374 GB，2026-09-27）：

| 项 | GB |
|---|---|
| Ubuntu + conda 环境 | 约 15 |
| 元数据 + 地图 | 17 |
| navtest 相机 | 121 |
| metric cache + LTF 权重 | 5 |
| navtrain 第 1 包（解压后） | 约 75 |
| 训练特征缓存 + 4 个 ckpt | 约 10 |
| **合计** | **约 245**（下载期间另有至多 55 GB 压缩包暂存） |

## 4. 验收（手册 §1、§3）
```bash
cd ~/E2E_planner && source ~/navsim_workspace/env.sh && conda activate navsim && export PYTHONPATH=$PWD
python tests/test_negaug_agent.py        # PASS，hit_rate 0.667，max_neg_active_frac > 0（与 docs/env.md cu128 列一致）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $CN_PROJECT_ROOT/configs train_test_split=navtest_dryrun \
  agent=ltf_camera_only agent.checkpoint_path=$NAVSIM_EXP_ROOT/checkpoints/ltf/ltf_seed_0.ckpt worker=single_machine_thread_pool experiment_name=dryrun
# Final average score ≈ 0.979
```

## 5. 最小验证实验
照 `docs/minimal_experiment_runbook.md` §5–§8 执行，命令无需改动。

## 本机注意事项
- **显存与 Windows 桌面共用**：这块 GPU 同时驱动显示器，占用 GPU 的程序（浏览器硬件加速、ChatGPT 桌面端等）会吃掉几百 MB。训练前关掉；`BS=32 ACC=2` 峰值约 6.4 GB，溢出就用 `BS=16 ACC=4`。
- **内存 32 GB**：WSL 上限 24 GB。评估保持 `worker.max_workers=4`；训练 OOM 时先把 `run_negaug_finetune.sh` 里的 `num_workers=6` 降到 4。
- **行尾**：仓库带 `.gitattributes`（`eol=lf`），Windows 端检出的脚本也是 LF。若见到 `$'\r': command not found`，说明是旧检出，见 `bootstrap_wsl.sh` 第 0 步提示。
