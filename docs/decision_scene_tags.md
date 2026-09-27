# 决策记录：场景标签获取路径（Step 0'，2026-09-14）

## 预检结果
- 脚本：`scripts/precheck_scene_tags.py`，检查 navtest 元数据 40 个日志、18,977 帧。
- 帧字典键：token, frame_idx, timestamp, log_name, log_token, scene_name, scene_token, map_location, roadblock_ids, vehicle_name, can_bus, lidar_path, lidar2ego_*, ego2global_*, ego_dynamic_state, traffic_lights, driving_command, cams, sample_prev, sample_next, anns, occ_gt_final_path, flow_gt_final_path。
- **不存在任何 scenario / tag / type 字段** → 路径 A（读取 OpenScene 元数据）不可行。

## 决策
| 路径 | 状态 | 理由 |
|---|---|---|
| A 元数据字段 | ✗ 关闭 | 预检证实字段不存在 |
| B nuPlan DB 回连 | ⏸ 暂缓 | 需 nuPlan test 分片数据库约 89 GB（第三方统计），本机剩余 159 GB 且需预留缓存空间；可在实验室服务器上作为交叉验证执行 |
| C 规则检测器 | ✓ 主路径 | 只依赖已下载的地图与标注：路口 lane connector 无信号灯 + 航向变化 ≥ 45°（无保护转向）；lane id 变化 + 目标车道间隙（汇入） |

## 对计划的影响
- Step 2 以路径 C 为主，人工抽检 50 帧精度 ≥ 85% 作为验收；风险 R5 触发，缓解措施已执行。
- 若后续在服务器上下载 nuPlan DB，用路径 B 对 C 的结果做一次精度回测并写入本文件。

## navtest 全量挖掘结果（路径 C，2026-09-14，第 3 版规则）
| 子集 | 场景数 | 定义 |
|---|---|---|
| navtest 全部（有路线） | 12,146 | 官方 token 12,282，136 个因无路线/帧不足被 SceneLoader 过滤 |
| unprotected_turn | 802（6.6%） | 未来 4 s 经过无信号灯 LEFT/RIGHT connector（几何推断）且航向变化 ≥ 45°；LEFT 488 / RIGHT 314（近似） |
| merging | 89（0.7%） | 同 roadblock 换道且目标车道 30 m 内有车 64 + 从停车区/车道边缘重新起步 25 |
| unprotected_crossing（候选第三类） | 205（1.7%） | t0 静止，经过无信号灯 STRAIGHT connector，30 m 内有运动车辆（无信号灯路口等待横向车流后直行，属间隙接受） |
| interact（三类并集） | 1,093 | 覆盖 92 / 136 个日志 |

### 规则迭代记录（由 BEV 抽检驱动）
1. v1：reentry = t0 静止 → 4 s 后 > 3 m/s 且附近有运动车 → 471 条，抽检发现多为堵车排队起步 → 否决。
2. v2：加“近 PUDO / 横向偏离 > 1.2 m / 车道 id 变化” → 244 条，抽检发现车道 id 变化多为直行穿越路口（lane → connector）→ 否决。
3. v3：车道 id 变化限定同一 roadblock → 25 条。merging 合计 89，规模偏小（触发风险 R4）。

### 对计划的影响
- **R4 触发**：navtest 汇入子集 < 500。缓解：(a) 训练用 navtrain 挖掘（约 8 倍规模，进行中）；(b) 评估层面把 unprotected_crossing 纳入交互子集并单列报告；(c) 汇入结果用 bootstrap CI 且明确标注样本量。
- 建议向导师提出把“无信号灯路口直行穿越”作为第三目标场景：它与无保护转向同属间隙接受问题，且 nuPlan 中样本量是汇入的两倍以上。
- 待人工核对：`results/review_sample_60_navtest.csv`（三类各 20）+ `results/review_png/`，标注模板 `results/review_png/review_labels_template.csv`。

产出：`configs/scene_filter/navtest_{unprotected_turn,merging,unprotected_crossing,interact}.yaml`。

## navtrain 全量挖掘结果（同一检测器 v4，2026-09-14）
| 子集 | 场景数 | 占比 | 覆盖日志 |
|---|---|---|---|
| navtrain 全部（有路线） | 103,288 | — | 1,192 |
| unprotected_turn | 6,057 | 5.9% | 561 |
| merging | 1,079 | 1.0% | 305 |
| unprotected_crossing | 2,671 | 2.6% | 543 |
| interact（并集） | 9,738 | 9.4% | 821 |

- 训练子集 ≥ 3,000 的验收条件满足（计划 Step 2 Go 判据）。
- 产出：`configs/scene_filter/navtrain_{unprotected_turn,merging,unprotected_crossing,interact}.yaml`；抽检样本 `results/review_sample_navtrain.csv`。
- 下一步：对 `navtrain_interact` 做 metric cache → 用 `scripts/build_negatives.py --split navtrain` 生成训练用负样本。

## 参考值：人类专家在 navtest 交互子集上的 PDM 子分数（由 v5 表回放得到，comfort/DDC 未记录）
| 层 | 场景 | NC | DAC | TTC | EP | PDMS |
|---|---|---|---|---|---|---|
| interact 全部 | 1,093 | 100 | 100 | 100 | 84.7 | 93.6 |
| unprotected_turn | 802 | 100 | 100 | 100 | 84.2 | 93.4 |
| merging | 89 | 100 | 100 | 100 | 86.8 | 94.5 |
| unprotected_crossing | 205 | 100 | 100 | 100 | 85.8 | 94.1 |
| expert_conservative（EP < 0.7） | 205 | 100 | 100 | 100 | 61.1 | 83.8 |

官方全量 navtest 人类 PDMS 为 94.8 [9]；交互子集略低，主要由 EP 决定。`expert_conservative` 层的人类 EP 仅 61.1，是评估“模型能否比人类更果断且同样安全”的关键分层。脚本：`eval/stratified_pdms.py`。
