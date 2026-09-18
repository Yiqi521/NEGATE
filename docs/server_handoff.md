# 服务器交接清单（训练与全量评估）

本机（8 GB GPU / 单盘）只完成 Step 0–3 与损失模块；以下工作需在实验室服务器（≥ 24 GB 显存，≥ 1.2 TB 磁盘）执行。

## 1. 需要复制到服务器的内容
| 内容 | 路径（本机） | 大小 | 说明 |
|---|---|---|---|
| 项目代码 | `~/E2E_planner/`（不含 `results/*.png`） | < 50 MB | 含 `conservative_negatives/`、`configs/`、`scripts/`、`docs/` |
| 负样本标签（navtest） | `results/labels/negatives_navtest.parquet`（用 `is_negative_strict` 列） | ~1 MB | 评估用（v13，含 O7） |
| 负样本标签（navtrain） | `results/labels/negatives_navtrain.parquet`（用 `is_negative_strict` 列） | ~5 MB | 训练用（v5：20,799 strict / 5,819 场景） |
| 场景标志表 | `results/labels/scene_flags_{navtest,navtrain}.parquet` | ~8 MB | 分层评估 |
| metric cache | `~/navsim_workspace/exp/metric_cache`（navtest 3.1 GB）、`metric_cache_navtrain_interact`（生成中） | 3–6 GB | 可在服务器重新生成（navtest 约 25 min / 22 线程） |
| 环境定义 | `~/navsim_workspace/env.sh`、navsim v1.1 commit 3e8291b | — | 见 `docs/env.md` |

## 2. 服务器上需要下载的数据
- navtrain 传感器 blob：445 GB（`download/download_navtrain.sh`，含历史帧）或 300 GB（仅当前帧）
- navtest 传感器 blob：223 GB（`download/download_test.sh`；本机只有相机部分）
- 地图 + 元数据：同本机（约 20 GB）

## 3. 服务器上的执行顺序（按研究计划 §16.2 的 A–G 组）

前提：最小验证实验（`docs/minimal_experiment_runbook.md`）四条标准全过。

```bash
# 0) 环境（任选：navsim = cu117 Ampere/Ada；navsim-cu128 = Blackwell 亦兼容 Ampere/Ada）
docker build -t negate:cu128 $PROJ    # 或 conda env create -f environment-cu128.yml
# 1) 数据：元数据+地图 17 GB；navtest 223 GB（相机+激光雷达）；navtrain 300 GB（仅当前帧）；metric cache 可从本机复制
# 2) 缓存：navtest metric cache；TransFuser 训练特征缓存（全量 navtrain，含激光雷达，约 4–8 h CPU）
# 3) 验收：官方 TransFuser 权重 navtest PDMS 84.0 ± 1.0（cu128 环境下重跑三种子作为正式基线）
```

| 组 | 内容 | 命令要点 | 次数 |
|---|---|---|---|
| A 从头训锚点 | TransFuser（含激光雷达）官方配置全量 100 epoch：基线 3 种子 + 最优 λ 3 种子 | `run_training.py agent=transfuser_agent`（基线）；`agent=tf_negaug agent.host_checkpoint=null agent.lambda_neg=<best>`（处理组，需新建 `configs/agent/tf_negaug.yaml`：host 换为 `TransfuserAgent`，其余同 ltf_negaug） | 6 |
| B λ 筛选 | 交互富集子集约 3 万帧（`build_finetune_subset.py --bg-ratio 2` 在全量 navtrain 上），微调 15 epoch，λ ∈ {0.05, 0.1, 0.2, 0.3} 单种子 | `run_negaug_finetune.sh ft_lam<λ> <λ>`，`SPLIT=navtrain_full_ft EPOCHS=15 BS=64 ACC=1` | 4 |
| C 补种子 | 最优与次优 λ 各补 2 种子 | 加 `seed=<s>` | 4 |
| D 四个对照组 | λ=0；错位（`agent.shuffle_seed=<s>`）；随机负样本（`agent.label_path=results/labels/negatives_navtrain_control_random.parquet`）；安全负样本（`..._control_safety.parquet`）；各 3 种子 | 同 B 的脚本 | 12 |
| E 三维消融 | 有无 O7（`label_path` 换 v12 标签）、`agent.loss_kind=infonce`、`agent.label_col=is_negative`（非严格） | 单种子筛 + 胜出补 2 种子 | 5 |
| F 第二宿主 PLUTO | nuPlan 验证集交互场景微调（λ=0 / 最优 λ / 错位，各 3 种子），Test14 闭环评估 | 见 §4 PLUTO 包装 | 9 |
| G 评估 | 每个 ckpt：`run_pdm_score`（navtest 全量）+ `behavior_metrics.py` + `stratified_pdms.py --baseline <λ=0 组> --behavior ...` | 见 §7 | 约 40 |

## 4b. PLUTO 宿主包装的最小实现范围（Step 7）
PLUTO 运行在 nuPlan devkit（矢量输入），NAVSIM 训练循环期望 `AbstractAgent` 接口。最小包装 `PlutoHostAgent(AbstractAgent)`：
- `get_sensor_config` → `SensorConfig.build_no_sensors()`（不读传感器）。
- `get_feature_builders` → 一个 `PlutoFeatureBuilder`：从 NAVSIM `AgentInput`/`Scene` 的地图 API 与标注框构造 PLUTO 的输入张量（自车历史、他车历史、地图折线、路线）。这是主要工作量（约 2–3 天），可复用 PLUTO 仓库 `feature_builder` 的几何代码，输入换成 NAVSIM 的 `Scene`。
- `get_target_builders` → PLUTO 的目标（专家未来轨迹等）；适配器再追加 `NegativeTargetBuilder`（不变）。
- `forward` → 调 PLUTO 模型，输出 dict 中放 `"trajectory"` (B,8,3)（PLUTO 输出 8 s @ 10 Hz，取前 4 s 每 0.5 s 采样）。
- `compute_loss` → PLUTO 原损失（含其 CIL 对比项）；适配器叠加分离损失。`lambda_scale` 取 PLUTO 轨迹回归项的权重。
- 闭环评估仍用 nuPlan devkit 的 `run_simulation.py`，加载微调后的 PLUTO 权重（从适配器 ckpt 中取 `host.*`）。
替代路线（若包装工作量超预算）：直接在 PLUTO 仓库内实现 `NegativeTargetBuilder` 与损失叠加（约 1 天），但这样"适配器通用"的主张只在 NAVSIM 侧成立，需在论文中说明。

## 4. 训练接入：通用适配器（已实现并通过单元测试，2026-09-18）

代码：`conservative_negatives/train/negaug_agent.py`；配置：`configs/agent/ltf_negaug.yaml`；启动脚本：`scripts/run_negaug_finetune.sh`。

```
NegativeAugmentedAgent(host, label_path, lambda_neg, loss_kind, k, label_col, shuffle_seed, host_checkpoint, lr, ...)
  get_sensor_config / get_feature_builders / forward / compute_trajectory / get_training_callbacks → 转发宿主
  get_target_builders → 宿主的 + NegativeTargetBuilder（按 token 查 NegativeBank → negatives (K,8,3), neg_mask (K,)）
  compute_loss        → 宿主损失 + λ(step)·SeparationLoss(pred["trajectory"], targets["trajectory"], negatives, neg_mask)
  get_optimizers      → 宿主参数 + 可覆盖学习率（微调用 2e-5）
  get_training_callbacks 追加 NegStatsCallback：把 loss_neg / neg_active_frac / lambda 写入 Lightning 日志
```
- **宿主代码零改动**；换宿主只改 `host:` 一段配置。PLUTO 需先包成 NAVSIM `AbstractAgent` 接口（Step 7）。
- 负样本以 target 形式进入 NAVSIM 缓存（`<cache>/<log>/<token>/negative_targets.gz`），与特征一起复用；标签文件或 K 变化需 `force_cache_computation=true`。
- 对照组：`agent.lambda_neg=0`（无分离损失）、`agent.shuffle_seed=<int>`（错位负样本，缓存名 `negative_targets_shuffled`）。
- 微调起点：`host_checkpoint` 指向官方权重；Lightning 产出的 checkpoint 键为 `agent.host.*`，适配器的 `load_state_dict` 已处理，可直接被 `run_pdm_score.py agent=ltf_negaug agent.checkpoint_path=<ckpt>` 装载。
- 单元测试结果（navtest 12 场景）：权重装入 missing 0 / unexpected 0；命中率 8/12 与标签一致；预训练 LTF 上 hinge 激活率 0.14–0.17（非零，R6 初步排除）；错位与 λ=0 分支正常；state_dict 回装 0 缺失。

**λ 的量级**：宿主 `transfuser_loss` 的轨迹项带权重（见 `TransfuserConfig.trajectory_weight`），分离损失与轨迹 L1 同量纲。λ 应按轨迹项权重解读：`lambda_neg` 是相对宿主轨迹权重的比例，脚本内换算，见 §16.1。

## 5. 验收（对应研究计划 §6.4 / §7.4）
- 至少一组 λ_neg 在 `navtest_interact` 子集上 EP 提升，且 NC / DAC / TTC 的 bootstrap 95% CI 下界 ≥ 基线均值。
- 3 种子；结果分层报告：全量 / turn / merging / crossing / `expert_conservative` 帧。

## 6. 实验追踪（Step 4 训练时使用）

`conservative_negatives/tracking.py` 零依赖，直接可用；若服务器已装 W&B 或 MLflow，传参即自动镜像，调用代码不变。

```python
from conservative_negatives.tracking import Run

with Run(f"tf_lambda{lam}_seed{seed}",
         config={"lambda_neg": lam, "seed": seed, "loss": "hinge", "label_col": "is_negative_strict"},
         tags=["step4", "transfuser"],
         wandb_project="negate") as run:        # wandb_project / mlflow_uri 可省略
    for step, batch in enumerate(loader):
        ...
        run.log({"loss": loss.item(), "loss_neg": l_neg.item(),
                 "neg_active_frac": stats["neg_active_frac"]}, step=step)
    run.finish({"pdms": pdms, "ep": ep, "unnecessary_stop_rate": usr})
```

每次运行落盘到 `results/runs/<run_id>/`（meta.json 含 git commit 与是否 dirty、conda 环境、GPU 型号、完整配置快照；metrics.jsonl 为逐步指标；summary.json 为最终指标），并在 `results/runs/index.csv` 追加一行便于横向比较。TensorBoard 曲线在 `results/runs/<run_id>/tb`。

**风险 R6 的自动检查**：训练日志中 `neg_active_frac` 若在 warm-up 结束后持续 < 1e-3，说明分离损失零激活，需改用提议打分型基线。

## 7. 评估流程（服务器上按序执行）

```bash
# 1) PDMS（官方）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $PROJ/configs \
  train_test_split=navtest agent=<agent> agent.checkpoint_path=$CKPT worker.max_workers=<N> \
  experiment_name=<exp>
# 2) 行为指标（三个自定义指标 + 人类参照组）
python eval/behavior_metrics.py --negatives results/labels/negatives_navtest.parquet \
  --agent-config configs/agent/<agent>.yaml --checkpoint $CKPT --out results/eval/behavior_<exp>.parquet --track
# 3) 合并报告：分层 PDMS + bootstrap CI + 配对差异 + 拒绝准则 + 行为指标
python eval/stratified_pdms.py --runs results/eval/<exp>_seed{0,1,2}.csv --baseline <baseline csv...> \
  --flags results/labels/scene_flags_navtest.parquet --negatives results/labels/negatives_navtest.parquet \
  --behavior results/eval/behavior_<exp>_seed{0,1,2}.parquet \
  --behavior-reference results/eval/behavior_human_navtest.parquet --out results/eval/<exp>_stratified.csv
```
