# 客观分类标准（替代人工标注）

> 目的：把研究计划中依赖人工审查（κ ≥ 0.6）的两处判断改为公式化、可复现的客观判据，并用**判据间一致性**（inter-criterion agreement）替代**评审者间一致性**。人工审查只保留为对判据分歧样本的小规模审计。
> 文献与阈值的核实状态见 §4（待补），实证比较见 §3。

## 1. 两个分类问题
- **P1 场景类别**：无保护转向 / 城市汇入 / 无信号灯直行穿越。已完全由地图与轨迹几何决定（`rule_detector.py`）：lane connector 首尾航向差（±30° 分左右转）、connector 是否受信号控制（地图 `has_traffic_lights()` ∨ 帧级 `traffic_lights` 列表）、同 roadblock 内 lane id 变化（换道）、上下客区 / 车道边缘起步。**不含任何人工判断**；人工抽检只用于验证实现正确性（已做 4 轮）。
- **P2 候选轨迹是否"安全但无理由地保守"**：这是原计划中最主观的部分。下面用交通工程与形式化安全领域的标准公式定义。

## 2. P2 的客观判据体系

| 编号 | 判据 | 公式 / 规则 | 参数（默认，待文献固定） | 实现 |
|---|---|---|---|---|
| O1 安全（回放） | NC = DAC = TTC = 1（NAVSIM PDM 评分器，非反应式 4 s 回放） | 官方 PDMS 子指标 | — | `pdm_eval.py` |
| O2 安全（冲突点） | 候选轨迹与每辆交汇他车在冲突点的 **PET**（后侵入时间）≥ τ_PET | PET = \|t_ego(冲突点) − t_agent(冲突点)\|，他车用日志真实未来位置 | τ_PET = 1.5 s | `objective_criteria.pet_of_candidate` |
| O3 安全（跟驰） | 沿候选轨迹对同向前车满足 **RSS 纵向安全距离** | d ≥ v_r·ρ + ½a_acc·ρ² + (v_r+ρ·a_acc)²/(2a_min,brake) − v_f²/(2a_max,brake) | ρ=1.0 s, a_acc=3.5, a_min,brake=4, a_max,brake=8 m/s² | `rss_follow_ok` |
| O4 进度显著更低 | EP(候选)/EP(专家) ≤ r_EP | NAVSIM EP（相对 PDM-Closed 上界） | r_EP = 0.8 | `build_negatives.py` |
| O5 非平凡 | EP 比 ≥ r_min | — | r_min = 0.3 | 同上 |
| **O6 无正当谨慎理由（间隙接受）** | 场景级：**不存在**与自车路径交汇的他车流，**或** t0 时已存在长度 ≥ **t_c（HCM 临界间隙）** 的可接受间隙 | 间隙 = 冲突点处相邻他车到达时刻之差（10 s 观测窗，截断区间取下界） | t_c：左转 4.1 s / 右转 6.2 s / 直行穿越 6.5 s / 支路左转 7.1 s（HCM，待核实）；默认 6.5 | `gap_analysis` |
| O7 专家保守性（间隙接受） | 专家四分类：no_conflict / decisive（未拒绝任何间隙）/ justified_wait（拒绝的间隙全部 < t_c）/ over_conservative（拒绝过 ≥ t_c 的间隙） | 同上 | 同上 | `objective_labels.py` |
| O8 起步延迟 | 专家在 ≥ t_c 间隙已开放时起步延迟 > t_startup | HCM 起步损失时间 | t_startup = 2.0 s | 同上 |

**负样本标签（v6）** = O1 ∧ O2 ∧ O3 ∧ O4 ∧ O5 ∧ O6。原 C3（回放 + 恒速外推间隙代理）保留为 `c3_replay` 字段供对照。

设计要点：
1. 他车未来位置取自日志（10 s，20 帧），不是恒速外推；观测窗必须 ≥ t_c，否则永远观测不到临界间隙（这是原 4 s 设计的盲点）。
2. 截断处理：窗口起点 / 终点处的间隙只知道下界；用下界做 "≥ t_c" 判定仍然成立，用作 "< t_c" 判定时标记为不确定。
3. 同向跟驰车辆（在 ≥ 80% 帧内都贴近自车路径）不算穿越冲突，交给 O3（RSS）处理。

## 3. 实证比较（navtest 交互子集，1,045 个可用 10 s 窗口的场景，2026-09-14）

### 3.1 专家行为四分类（O7）
| 类别 | no_conflict | decisive | justified_wait | over_conservative（t_c=6.5） | over_conservative（t_c=4.1） |
|---|---|---|---|---|---|
| unprotected_turn (776) | 480 | 171 | 104 | 5 | 13 |
| merging (87) | 34 | 22 | 30 | 1 | 3 |
| unprotected_crossing (198) | 42 | 56 | 91 | 9 | 9 |
| 合计 | 556 (53%) | 249 (24%) | 225 (22%) | 15 (1.4%) | 25 (2.4%) |

- 人类专家在 navtest 中**几乎不拒绝可接受间隙**（1.4–2.4%），起步延迟 > 2 s 且有间隙的情形为 0。人类的"保守"主要表现为进度偏低而非拒绝间隙，这与基线结果中 `expert_conservative` 层（EP 相对 PDM-Closed < 0.7）的定义是两个不同侧面，论文中应分别报告。
- 22% 的场景是**有理由的等待**（存在冲突流且没有可接受间隙）——这些场景中生成的"更快"候选不应被标为过度保守的反例（即候选本身也不该被推离）。

### 3.2 客观判据 vs 回放式 C3（v5）
| | C3 通过（无理由） | C3 不通过（有理由） |
|---|---|---|
| O6 无理由 | 701 | 29 |
| O6 有理由 | **277** | 38 |

- 一致率 70.7%，**κ = 0.10**（t_c = 4.1 时 κ = 0.11）。分歧几乎全部是"O6 判有理由、C3 判无理由"的 277 个场景：非反应式回放中，日志里的他车已经为真实（更谨慎的）自车让过路，所以更快的参考轨迹极少触发碰撞或 TTC。
- 在 O7 = justified_wait 的 225 个场景中，C3 有 201 个（89%）仍然通过。**结论：回放式 C3 系统性地漏判有理由的谨慎；间隙接受判据是必要的补充，且有交通工程文献支撑。**

### 3.3 候选安全（O2/O3）
- 代表性候选（静止起步 → G1 延迟 1.5 s；行驶中 → G2 ×0.65）：O2 ∧ O3 通过率 93.5%；256 个存在交汇的候选中 PET 5% 分位 1.0 s、中位 3.25 s。
- 与回放 C1（NC/DAC/TTC）互补：PET 捕捉"恰好错过"的低裕度穿越，RSS 捕捉跟驰距离不足。

## 4. 文献与阈值来源（2026-09-14 在线核实；"部分核实"表示数值来自二手来源）

### 4.1 间隙接受与临界间隙（O6 / O7 / O8）
| 参数 | 值 | 来源 | 核实 |
|---|---|---|---|
| HCM 基准临界间隙 t_c,base | 主路左转 **4.1** s；支路右转 **6.2**（四车道主路 6.9）；支路直行 **6.5**；支路左转 **7.1**（四车道 7.5） | TRB, *Highway Capacity Manual*, 6th ed. (2016) Vol. 3 Ch. 20 TWSC（方法同 HCM 2010 Ch. 19、HCM 7th 2022）；数值经 Roess & Prassas, *The HCM: A Conceptual and Research History Vol. 2*, Springer 2020, DOI 10.1007/978-3-030-34480-1 附录 A 核对 | 已核实 |
| HCM 基准跟进时距 t_f,base | 2.2 / 3.3 / 4.0 / 3.5 s（同上四类运动） | 同上 | 已核实 |
| 修正式 | t_c = t_c,base + t_c,HV·P_HV + t_c,G·G − t_c,T − t_3,LT | HCM 2000 Eq. 17-1/17-2，结构沿用至 6th | 已核实（部分修正系数未核实） |
| 临界间隙定义 | Raff：接受间隙 CDF 与拒绝间隙补 CDF 的交点；Troutbeck ML：每位驾驶员 r_d < t_c,d < a_d，对数正态 t_c 极大似然 | M. S. Raff & J. W. Hart, *A Volume Warrant for Urban Stop Signs*, Eno Foundation, 1950；R. J. Troutbeck, QUT Res. Rep. 92-5, 1992；R. J. Troutbeck, "Revised Raff's Method for Estimating Critical Gaps," *TRR* 2553:1–9, 2016, DOI 10.3141/2553-01；W. Brilon, R. Koenig, R. J. Troutbeck, "Useful estimation procedures for critical gaps," *Transp. Res. A* 33(3–4):161–186, 1999, DOI 10.1016/S0965-8564(98)00048-2 | 已核实 |
| 经验接受曲线（无保护左转，LTAP-OD） | 接受率 15% / 50% / 85% 对应间隙 **4.1 / 6.0 / 8.6 s**；< 3 s 全拒，> 12 s 全接受；logistic 拟合 | D. R. Ragland, S. Arroyo, S. E. Shladover, J. A. Misener, C.-Y. Chan, "Gap acceptance for vehicles turning left across on-coming traffic," TRB 85th Annual Meeting, 2006（UC Berkeley PATH 报告） | 已核实（全文） |
| 个体决策模型 | 风险–收益权衡下的个体临界间隙 | M. A. Pollatschek, A. Polus, M. Livneh, *Transp. Res. B* 36(7):649–663, 2002, DOI 10.1016/S0191-2615(01)00024-8 | 已核实 |
| 起步损失时间 | **2.0 s**（HCM 信号交叉口默认） | HCM 6th/7th Ch. 19 | 已核实（数值） |
| 感知–制动反应时间 | 设计值 2.5 s（AASHTO）；期望信号 0.7–0.75 s，常见非预期 ≈ 1.25 s，突发 ≈ 1.5 s | AASHTO *Green Book* 6th/7th ed.；M. Green, *Transportation Human Factors* 2(3):195–216, 2000, DOI 10.1207/STHF0203_1 | 已核实 |

**采用值**：t_c 按运动类型取 HCM 基准；无保护左转另用 Ragland 8.6 s 作为"强过度保守"（拒绝 85% 人类会接受的间隙）标记、4.1 s 作为"合理谨慎"下界；O8 起步延迟阈值 t_startup = 2.0 s（HCM），敏感性分析 4.5 s（2.0 + AASHTO 2.5）。

### 4.2 替代安全指标（O2 / O1 的补充）
| 指标 | 定义 / 阈值 | 来源 | 核实 |
|---|---|---|---|
| PET | 后侵入时间 = 侵入结束到冲突车到达潜在碰撞点的时间（T4 − T2）；原文无数值阈值 | B. L. Allen, B. T. Shin, P. J. Cooper, "Analysis of Traffic Conflicts and Collisions," *TRR* 667:67–74, 1978 | 已核实（全文） |
| PET 阈值 | PET ≤ 1 s 与对向左转事故关联最强 | P. Peesapati, M. Hunter, M. Rodgers, *TRR* 2386:42–51, 2013 | 部分核实 |
| TTC 阈值 | 1.5 s（瑞典冲突技术早期版本 TA < 1.5 s；1987 版改用 TA–CS 严重度图，严重冲突 = 等级 ≥ 26） | C. Hydén, Bulletin 70, Lund Inst. Tech., 1987；A. Laureshyn & A. Várhelyi, *Swedish TCT Observer's Manual* v1.1, 2020 | 部分核实 |
| TTC 扩展量 TET / TIT | TET = TTC < TTC* 的累计时间；TIT = ∫(TTC* − TTC)dt | M. M. Minderhoud & P. H. L. Bovy, *AAP* 33(1):89–97, 2001, DOI 10.1016/S0001-4575(00)00019-1 | 已核实 |
| DRAC 阈值 | 3.4 m/s²（AASHTO）；3.35 m/s²（Archer 2005） | Y. Kuang et al., *PLoS ONE* 10(9):e0138617, 2015, DOI 10.1371/journal.pone.0138617；J. Archer, PhD thesis, KTH, 2005 | 已核实 / 部分核实 |
| 车头时距 | < 1 s 潜在危险，< 2 s 有风险 | K. Vogel, *AAP* 35(3):427–433, 2003, DOI 10.1016/S0001-4575(02)00022-2 | 已核实（引文） |
| 综述 | 38 个邻近性指标及阈值；AD 领域形式化定义 | S. M. S. Mahmud et al., *IATSS Research* 41(4):153–163, 2017, DOI 10.1016/j.iatssr.2017.02.001；L. Westhofen et al., *Arch. Comput. Methods Eng.*, 2022, DOI 10.1007/s11831-022-09788-7 | 已核实 |

**采用值**：τ_PET = 1.5 s（主），敏感性 1.0 / 2.0 s；TTC* = 1.5 s 用于 TET；候选轨迹的 DRAC 由 PDM Comfort 子项间接覆盖，不单独门控。

### 4.3 形式化安全包络（O3）
| 项 | 内容 | 来源 | 核实 |
|---|---|---|---|
| RSS 纵向安全距离 | d_min = [v_r ρ + ½ a_acc ρ² + (v_r + ρ a_acc)²/(2 a_min,brake) − v_f²/(2 a_max,brake)]₊（Lemma 2）；路口 / 不同路线几何的纵向排序与优先权规则（Def. 14–18） | S. Shalev-Shwartz, S. Shammah, A. Shashua, "On a Formal Model of Safe and Scalable Self-driving Cars," arXiv:1708.06374, 2017（v6 2018） | 已核实（全文） |
| 参数默认值 | ρ_ego = 1 s，ρ_other = 2 s，a_acc,max = 3.5，a_brake,min = 4，a_brake,max = 8 m/s²（"仅为建议"） | Intel/Mobileye *ad-rss-lib*, "Appendix – Parameter Discussion" | 已核实 |
| RSS 的保守性实证 | 高速跟驰中 RSS 过度保守、降低效率 | O. Hassanin, X. Wang, X. Wu, X. Xu, *AAP* 177:106799, 2022 | 部分核实 |
| 运动学假设标准 | 对其它道路使用者的 v_max / a_max / β / ρ 假设类别（含被遮挡使用者），数值由开发者 / 监管选定；法规先例：UNECE R157 减速 > 5.0 m/s² 为紧急操作；德国 AV 法规前车最大减速 10 m/s² | IEEE Std 2846-2022；IEEE Std 3321-2024 | 部分核实（正文付费） |

**采用值**：本实现 `RSSParams(rho=1.0, 3.5, 4.0, 8.0)` 与 ad-rss-lib 一致。

### 4.4 遮挡下的合理谨慎（O6 的补充，待实现）
| 项 | 内容 | 来源 | 核实 |
|---|---|---|---|
| 幻影车可达性 | 未观测车道段上以 v ∈ [0, v_max] 恒速传播的幻影粒子；风险 = 粒子密度 | M.-Y. Yu, R. Vasudevan, M. Johnson-Roberson, *IEEE RA-L* 4(2):2235–2241, 2019, DOI 10.1109/LRA.2019.2900453 | 已核实 |
| 集合式验证 | 传感范围边界处的隐藏障碍，v ≤ 1.1·v_lim，a_max = 10 m/s²；自车占用与幻影占用相交则不安全 | P. F. Orzechowski, A. Meyer, M. Lauer, *IEEE ITSC* 2018, pp. 1729–1736 | 已核实 |
| 简化可达性量化 | 幻影车初始位置均匀分布于动态幻影集，速度均匀 [0, v_max]，可达质量分段式 | H. Park, J. Choi, H. Chin, S.-H. Lee, D. Baek, *IEEE RA-L*, 2023 (arXiv:2306.07004) | 已核实 |
| 本研究采用的判据（推导） | 谨慎有理由 ⇔ d_occ / (1.1·v_lim) < t_clear + ρ（ρ = 1 s） | 上述三文的确定性特例，非原文原句 | 推导 |

### 4.5 场景分类（P1）
| 项 | 内容 | 来源 | 核实 |
|---|---|---|---|
| 换道定义与时长 | 车道 ID 变化 + 车身边界穿越法定义时长；NGSIM 模态 ≈ 3 s（边界法）/ 5–6 s（曲率法）；自然驾驶均值 6.28 s | C. Thiemann, M. Treiber, A. Kesting, *TRR* 2088:90–101, 2008；T. Toledo & D. Zohar, *TRR* 1999:71–78, 2007；S. E. Lee, E. C. B. Olsen, W. W. Wierwille, NHTSA DOT HS 809 702, 2004 | 已核实 |
| "有车车道" | 目标车道前后车 THW < 2 s | Vogel 2003 | 已核实 |
| 路口控制类型 | Signalized / TWSC（含让行与无控制）/ AWSC / 环岛 | HCM 6th Ch. 19–22；MUTCD 11th ed. 2023 Part 2B | 已核实 |

**对应实现**：`rule_detector.py` 的换道 = 同 roadblock 内 lane id 变化（Thiemann 的 lane-ID 事件）；控制类型 = 地图 `has_traffic_lights()` ∨ 帧级信号灯记录（Signalized）/ STOP_SIGN、STOP_LINE、YIELD 层（TWSC / AWSC，待加入以区分主路与支路 t_c）。

### 4.6 一致性统计与收敛效度（替代人工 κ）
| 项 | 内容 | 来源 |
|---|---|---|
| κ 判读 | McHugh：< 0.60 不足，0.60–0.79 中等，0.80–0.90 强，> 0.90 近乎完美；Landis–Koch：0.41–0.60 中等，0.61–0.80 显著 | M. L. McHugh, *Biochemia Medica* 22(3):276–282, 2012, DOI 10.11613/BM.2012.031；J. R. Landis & G. G. Koch, *Biometrics* 33(1):159–174, 1977 |
| 多标注者 κ | Fleiss' κ | J. L. Fleiss, *Psychol. Bull.* 76(5):378–382, 1971 |
| 收敛效度 | 独立机制的方法测同一构念应一致 | D. T. Campbell & D. W. Fiske, *Psychol. Bull.* 56(2):81–105, 1959 |
| 弱监督合并 | 多个启发式标注函数按一致 / 分歧建模合并 | A. Ratner et al., "Snorkel," *PVLDB* 11(3):269–282, 2017 |

**采用做法**：把 O1（回放安全）、O2/O3（PET/RSS）、O6（HCM 间隙）、遮挡可达性（待实现）视为独立"算法标注者"，报告两两 Cohen κ 与 Fleiss κ，目标 κ ≥ 0.80（McHugh"强"）；分歧样本 ≤ 50 条做实现审计。

## 5. 对研究计划的修订
- Step 1 判据 C3 由"回放事后验证"改为 O6（间隙接受）+ O2/O3（候选冲突点与跟驰安全），回放 C1 保留。
- Step 3 验收中的"人工审查 ≥ 200 条、κ ≥ 0.6"改为：(a) 报告 O6 与回放 C3 的 κ 及分歧类型；(b) 对 O6/C3 分歧样本做 ≤ 50 条的**审计**（不是标注），只用于发现实现错误；(c) 阈值敏感性表（t_c ∈ {4.1, 6.5}，τ_PET ∈ {1.0, 1.5, 2.0}）。
- 新增专家四分类 O7 作为评估分层（decisive / justified_wait / over_conservative），与 `expert_conservative` 层并列。

## 6. 判据迭代记录
### v6（首次集成客观判据，navtest 1,045 场景）
- 场景级 O6 通过 69.9%（回放 C3 为 93.6%）；负样本 2,591 条，覆盖 703 场景（v5 同场景集：3,528 条 / 954 场景）。
- 相对 v5 移除 1,017 条：来自 justified_wait 691、decisive 244、no_conflict 47、over_conservative 35。
- **问题 1**：decisive 场景（专家未拒绝任何间隙即通过）被 O6 判为"有理由"，因为 t0 的间隙短于 t_c。但专家已安全通过（PET 达标），这在间隙接受理论中就是"被接受的间隙"，即可接受间隙存在。**修正（v7）**：O6 增加分支 `decisive ∧ PET_expert ≥ τ_PET`（行为揭示的可接受间隙）。
- **问题 2**：RSS 检查（O3）在回放安全的候选中误报 913 条。原因是非反应式伪影：候选比专家慢后，日志中原本跟在自车后面的车"出现"在候选前方，被当作前车。**修正（v7）**：只对"同一时刻位于专家前方"的车辆做 RSS 检查；`need` 下限从 1e-3 改为 0.5 m 避免除零放大。
- O2（PET < 1.5 s）只淘汰 62 条候选，符合预期（低裕度穿越少见）。

### v7（修正 decisive 分支与 RSS 前车过滤，navtest 1,045 场景）
- 场景级 O6 通过 76.4%：no_conflict 100%、decisive 94.4%、over_conservative 46.7%、justified_wait 0%（定义如此）。负样本 2,746 条，覆盖 761 场景（转向 624 / 760，汇入 43 / 87，直行穿越 94 / 198）。
- 候选级：回放安全的 10,027 条中 PET < 1.5 s 淘汰 62 条；RSS 仍淘汰 797 条，其中 514 条来自旧 G3 算子，且失败裕度比中位 0.77——排队场景中**专家自身**也常不满足绝对 RSS（RSS 假设后车先加速 ρ 秒，对低速排队偏严）。**修正（v8）**：RSS 改为相对判据——候选裕度 ≥ min(1, 专家裕度)。
- **判据间一致性（`eval/criteria_agreement.py`）**：
  - 场景级"无正当理由"：gap_HCM vs replay κ = 0.079（一致率 0.75）；expert_not_justified vs replay κ = 0.073；gap_HCM vs expert_not_justified κ = **0.940**。三者 Fleiss κ = 0.41。→ 回放式判据是离群者；两个基于间隙接受理论的判据高度收敛（Campbell–Fiske 意义上的收敛效度，注意二者共享冲突检测机制，独立性有限）。
  - 候选级"安全"：三判据真值比例均 > 0.92，κ 受基率效应压低（0.17 / −0.01 / −0.03）；改报告分歧绝对数：仅 PET 否 58、仅 RSS 否 793（v8 修正）、仅回放否 340。
- **v8 新增**：t_c 按运动类型与停车控制自动选取（左转 4.1 / 支路左转 7.1 / 右转 6.2 / 直行 6.5，STOP_SIGN / STOP_LINE / YIELD 12 m 内判为支路）；Ragland 8.6 s 作为"强过度保守"标记字段 `expert_strong_over_conservative`。
