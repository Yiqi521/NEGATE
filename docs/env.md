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
