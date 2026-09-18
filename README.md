# E2E_planner — 保守负样本训练框架（工程目录）

研究计划：[研究计划_SUYiqi_2026-09.md](研究计划_SUYiqi_2026-09.md)　环境记录：[docs/env.md](docs/env.md)　决策记录：[docs/decision_scene_tags.md](docs/decision_scene_tags.md)

## 目录
| 路径 | 内容 |
|---|---|
| `conservative_negatives/scenes/rule_detector.py` | Step 2 规则场景检测器（无保护转向 / 城市汇入） |
| `conservative_negatives/generators/operators.py` | Step 3 生成算子 G1–G4 与运动学过滤 |
| `conservative_negatives/filters/pdm_eval.py` | 任意轨迹 → PDMS 子分数（复用官方 PDM 评分器与 metric cache） |
| `conservative_negatives/agents/ltf_camera_only.py` | 纯相机 LTF 智能体（本机评估用） |
| `conservative_negatives/losses/` | 分离损失（hinge / InfoNCE）与 NegativeBank 数据模块 |
| `docs/server_handoff.md` | 服务器交接：需复制的数据、命令顺序、训练接入设计 |
| `docs/server_requirements.md` | 服务器申请用需求配置（最低 / 推荐、存储与算力预算、中英日简述） |
| `docs/results_baseline.md` | LTF 基线 3 种子分层结果与含义 |
| `docs/objective_criteria.md` | 客观分类标准：判据体系、文献阈值、实证比较、迭代记录 |
| `conservative_negatives/definition/objective_criteria.py` | 客观判据 O1–O6：间隙接受、PET、RSS、起步延迟（替代人工标注） |
| `conservative_negatives/definition/occlusion.py` | 客观判据 O7：遮挡可达性（幻影车），处理"看不见的来车方向" |
| `conservative_negatives/tracking.py` | 实验追踪：零依赖本地记录 + 可选 TensorBoard / W&B / MLflow 镜像 |
| `conservative_negatives/train/negaug_agent.py` | **通用适配器**：给任意 NAVSIM 智能体附加负样本目标与分离损失（宿主零改动） |
| `configs/agent/ltf_negaug.yaml` | 适配器 × LTF 的 Hydra 配置（λ、损失形式、错位对照、微调起点） |
| `results/labels/` | **稳定标签目录**（下游一律引用这里的固定文件名） |
| `configs/scene_filter/*.yaml` | 自动生成的子集 token 列表（可直接作 NAVSIM scene_filter） |
| `configs/agent/ltf_camera_only.yaml` | 纯相机 LTF 的 Hydra 配置 |
| `scripts/` | 可执行脚本，见下 |
| `results/` | 表格、日志、抽检样本（不存 checkpoint） |

## 环境
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate navsim
source ~/navsim_workspace/env.sh          # NAVSIM 环境变量
export PYTHONPATH=$PWD                    # 使 conservative_negatives 可导入
```

## 脚本
| 脚本 | 作用 | 依赖 |
|---|---|---|
| `scripts/precheck_scene_tags.py <log_dir>` | 检查日志 pickle 是否含场景类型字段 | 元数据 |
| `scripts/mine_scenes.py --split navtest --out results/scene_flags_navtest.parquet` | 规则场景挖掘 | 元数据 + 地图 |
| `scripts/build_negatives.py --split navtest --flags ... --subset interact --out ...` | 生成负样本候选并做 C1–C4 过滤 | metric cache |
| `scripts/render_review_sample.py` | 为人工抽检样本渲染 BEV 图 | 元数据 + 地图 |
| `scripts/eval_ltf_navtest.sh <seed>` | 官方 LTF 权重在 navtest 上的基线评估 | 相机 blob + metric cache |
| `scripts/build_finetune_subset.py` | 按本地已下载日志构建微调子集（交互 + 背景）并生成 scene_filter | 元数据 + 标签 |
| `scripts/run_negaug_finetune.sh <run> <λ> [shuffle]` | 最小验证实验：LTF 微调（batch 32 × 累积 2，10 epoch） | navtrain 传感器（1 个压缩包） + 标签 |
| `scripts/eval_ltf_all.sh` | LTF 3 seeds 全量评估 + 分层统计一键脚本 | 相机 blob + metric cache |
| `scripts/merge_shards.py` | 合并分片构建结果 | — |
| `eval/stratified_pdms.py --runs ... --flags ... [--baseline ...] [--behavior ...]` | 分层 PDMS、bootstrap CI、配对差异、拒绝准则、行为指标 | run_pdm_score 的 CSV |
| `scripts/objective_labels.py` | 专家四分类与客观特征（间隙、PET、RSS、起步延迟），10 s 窗口 | 元数据 + 地图 |
| `eval/criteria_agreement.py --negatives ...` | 判据间两两 Cohen κ / Fleiss κ（替代人工 κ） | build_negatives --objective 输出 |
| `eval/behavior_metrics.py` | 三个行为指标：不必要停车率、起步延迟 Δ、间隙接受率（`--human` 为参照组） | 相机 blob + 标签文件 |

## 官方命令（NAVSIM v1.1）
```bash
# navtest 指标缓存
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
# 评估（示例：本机纯相机 LTF）
bash scripts/eval_ltf_navtest.sh 0
```

## 当前状态（2026-09-14 晚）
- [x] Step 0 环境：navsim v1.1 + conda env + 地图 + navtrain/navtest 元数据 + navtest metric cache（12,146）
- [x] Step 0' 预检：路径 A 不可行 → 路径 C 规则检测器（4 轮由 BEV 抽检驱动的迭代）
- [x] Step 2 navtest 子集：unprotected_turn 802 / merging 89 / unprotected_crossing 205（候选第三类）/ 并集 1,093 → `configs/scene_filter/navtest_*.yaml`
- [x] Step 1 定义实现 + C2/C4 阈值校准（C2 ≤ 0.8, C4 ≥ 0.3）→ `conservative_negatives/definition/definition.md`
- [x] Step 3 流水线：G1–G4 + 运动学 + C1–C4（C3：行驶中 ×1.15 / 静止起步提前 1 s，参考只查 NC+TTC）；**正式标签** `results/labels/negatives_navtest.parquet`（v13：O1–O6 客观判据 + O7 遮挡可达性；`is_negative_strict` 2,783 条 / 742 场景）
- [x] Step 4 训练侧模块：`losses/separation_loss.py`（hinge / InfoNCE + warm-up）、`losses/negative_bank.py`（token → 负样本张量）；接入设计见 `docs/server_handoff.md`
- [x] navtrain 子集：turn 6,057 / merging 1,079 / crossing 2,671 / 并集 9,738 → `configs/scene_filter/navtrain_*.yaml`
- [x] navtrain 训练标签：v1 纯回放 `negatives_navtrain_v1.parquet`（30,251 / 8,487）；`results/labels/negatives_navtrain.parquet`（v5：strict 20,799 / 5,819 场景）+ metric cache `metric_cache_navtrain_interact`
- [x] 评估脚本 `eval/stratified_pdms.py`（已用人类专家分数验证）
- [x] LTF 纯相机评估预演：40 场景 / 21 s，流程正常（`configs/train_test_split/navtest_dryrun.yaml`）
- [x] navtest 相机 blob（121 GB，147 日志）就位于 `$OPENSCENE_DATA_ROOT/sensor_blobs/test`
- [x] LTF 基线（3 seeds）：全量 PDMS 83.5 ± 0.45（官方 83.8）；分层结果与含义见 `docs/results_baseline.md`
- [x] 客观判据替代人工标注：`scripts/build_negatives.py --objective`（HCM 间隙 + PET + 相对 RSS），一致性见 `eval/criteria_agreement.py`；人工只做分歧审计（可选）
- [ ] G3 算子重设计；专家保守性标记入表；训练接入（需服务器）
