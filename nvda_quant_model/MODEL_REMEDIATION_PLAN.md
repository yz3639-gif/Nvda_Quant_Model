# NVDA 模型整改方案

目标：提高模型精准度和及时反应能力，同时坚持所有改动必须经过回测验证。任何新模块不能只因为单段历史好看就进入主模型。

## 总原则

- 主模型优先稳定，不因短期收益更高就替换。
- 新候选必须经过 24/36/60 个月复验。
- 及时反应层只做预警或小权重 shadow overlay，必须证明不扩大回撤。
- 实时新闻、期权、买卖盘只能作为当前 overlay；没有 point-in-time 历史数据前，不进入历史训练。
- 每一轮优化都必须输出 CSV/JSON/Markdown 报告，方便复查。

## P0：反应池过拟合整改

问题：当前反应池参数来自同一段历史 sweep，可能二次过拟合。

解决方案：
- 用前段时间训练参数，后段时间验证参数。
- 训练集只负责选 `top_n / min_vote_count / prob_threshold / min_vote_weight`。
- 验证集只用于评估，不参与选参。

验收标准：
- 验证集 combined 年化不低于 baseline。
- 验证集 combined Sharpe 不低于 baseline。
- 验证集 combined 最大回撤不能比 baseline 差超过 2 个百分点。
- 反应层方向精度 > 60%，且 active days >= 8。

当前状态：已完成第一版。时间切分验证没有通过，反应池继续保留为观察层，不进入主交易信号。

## P1：候选晋级机制整改

问题：长跑搜索会产生大量漂亮但不稳定的规则。

解决方案：
- 固化 `candidate_promotion` 作为候选晋级流水线。
- 候选按收益、Sharpe、胜率、利润因子、样本数分桶抽取。
- 每个候选跑 24/36/60 个月复验并打标签。

验收标准：
- 只有 `promote_candidate` 能替换主模型。
- `watchlist_*` 只能进入反应池。
- `reject_*` 不参与任何交易决策。

当前状态：已完成第一版，未发现可替换主模型候选。

## P2：Meta Decision Layer

问题：主模型精准但慢，反应池快但样本少，实时事件层还没有统一决策出口。

解决方案：
- 新增 `meta_decision_layer`。
- 输入：主模型、反应池、新闻、期权、订单流、市场 regime。
- 输出：`NO_TRADE`、`WATCH`、`TACTICAL_LONG`、`BASELINE_LONG`、`BLOCK_LONG`、`RISK_OFF`。

验收标准：
- 每个决策状态必须可回测。
- `TACTICAL_LONG` 不允许降低主模型 Sharpe。
- `BLOCK_LONG` 必须证明能降低回撤或减少亏损交易。

当前状态：已完成第一版。`meta_decision_layer` 已统一 baseline、reaction、risk_off、实时新闻、期权快照与买卖盘接口。严格生产模式只允许已通过验证的层改变仓位；当前 reaction 未通过、risk_off 仍是 watchlist_only，所以正式输出保持 baseline-only，不降级。research/watchlist 模式在 36/60 个月有改善但 24 个月失败，不能升生产。

## P3：Point-in-time 事件数据整改

问题：NVDA 是事件驱动股票；没有历史时间戳数据，新闻/期权/财报无法严肃入模。

解决方案：
- 接入历史新闻时间线，至少保存 headline、source、published_at、sentiment、event_type。
- 接入历史期权 IV/skew/ATM straddle expected move。
- 建财报事件表：财报日、EPS surprise、revenue surprise、guidance surprise、盘后/盘前标记。

验收标准：
- 所有事件特征必须按 published_at 对齐，不能用未来数据。
- 提供 point-in-time cache。
- 事件特征加入模型后必须通过 walk-forward 回测。

当前状态：框架有，历史数据不足。

## P4：熊市/下行 regime 整改

问题：当前模型主要 long-only，遇到 2022 类型市场只能空仓防守，不能主动识别风险。

解决方案：
- 单独训练 `risk_off` 和 `bearish_regime` 检测。
- 先不做空，先做 `BLOCK_LONG`。
- 后续如有足够证据，再评估小仓位 short/hedge。

验收标准：
- 60个月回测中 2022 年亏损交易减少。
- 最大回撤下降。
- 不显著牺牲 2024/2025 牛市收益。

当前状态：已完成第一版。`risk_off_guard` 已回测 `macro_trend / peer_stress / high_vol_break / composite_risk / holiday_risk` 五类 BLOCK_LONG 规则。60个月窗口通过生产门槛，但24/36个月没有同时通过，且验证段阻断样本偏小，因此当前结论是 watchlist only，不进入正式交易层。

## P5：持续优化自动化

问题：此前长跑只是持续搜索，不是持续优化。

解决方案：
- 长跑搜索继续作为原料来源。
- 每次 checkpoint 后自动跑候选晋级。
- 晋级后自动跑反应池验证。
- 只有出现显著变化才通知。

验收标准：
- 不重复启动长跑进程。
- 每次优化都有报告。
- 主模型替换必须有明确证据链。

当前状态：已完成第二版。新增 `optimizer_monitor` 作为统一入口：检查 repaired 双轨进程、避免重复进程、缺失时续跑、发现 high-sample 新 qualified 候选时自动触发跨窗口复验，并写出 `outputs/optimizer_monitor_repaired/monitor_state.json`。heartbeat 已切换为调用该模块。

## P6：高样本量决策整改

问题：当前 strict best 只有 31 笔交易，最新全局 best 甚至只有 21 笔。它们可以说明规则“很准”，但样本量不足以支撑高置信决策。

解决方案：
- 新增 `high_sample_optimizer` 独立长跑线，不干扰 strict precision optimizer。
- 硬性加入 `sample_gate`：默认交易数 >= 60、active days >= 90。
- 再加入 `quality_gate`：方向精度、最差年份精度、Sharpe、年化、最大回撤、胜率、利润因子同时过线。
- 输出 `best_high_sample.json` 和 `best_qualified_high_sample.json`，把“样本足够”和“质量足够”分开看。

验收标准：
- 先找到 60+ trades 且质量达标候选。
- 再向 80/100 trades 推进，不能用明显更差的回撤或胜率换样本量。
- 任何高样本候选仍需进入 24/36/60 个月复验，不能直接替换主模型。

当前状态：第一版已完成并开始独立长跑。快速探针已找到 69 笔交易的合格候选：年化约 19.95%、Sharpe 1.41、最大回撤 -11.05%、胜率 55.07%、利润因子 2.04、方向精度 58.12%。该候选只证明高样本路线可行，尚未通过跨窗口晋级。

## P7：回测停机记账维修

问题：`BacktestEngine` 在触发最大回撤停机后，只把下一日目标仓位设为 0，但没有进入永久 halt 状态。对于长窗口高样本候选，后续信号仍可能被当作新入场处理，造成反复扣交易成本和虚假交易记录。这个问题会把 60M 结果放大得过分难看，尤其是 2022 触发 drawdown stop 后。

解决方案：
- `BacktestEngine` 增加 `halted` 状态。
- 触发最大回撤后关闭当前持仓，记录 `drawdown_stop`，后续所有目标仓位强制为 0。
- 增加单元测试，防止停机后假重入。
- 旧输出目录保留审计，但不再作为晋级依据；新长跑目录使用 `_repaired` 后缀。
- 行情缓存读失败时自动丢弃坏 CSV 并重下；写缓存改为临时文件原子替换，降低并发写坏文件概率。

验收标准：
- 停机后 equity curve 不再出现新仓位。
- 停机后 trades 不再出现假交易。
- 高样本候选必须在 repaired 输出下重新搜索，并重新跑 24/36/60 复验。

当前状态：已完成代码维修，测试通过。旧 long_run/high_sample 进程已停止；已启动 `long_run_optimizer_repaired` 和 `high_sample_optimizer_repaired` 两条新长跑。`high_sample_validation_repaired` 已跑通，并自动修复一个损坏的 NVDA cache 文件。

## P8：GitHub 同步纪律

问题：用户要求每个有效进步都自动更新到 GitHub，但自动同步不能牺牲安全性或把旁路研究资料混进 NVDA 主线。

解决方案：
- 只同步 NVDA 主线相关变更：`nvda_quant_model/`、NVDA 测试、根 README/.gitignore 等必要工程文件。
- 不同步密钥、`.env`、`data_source_credentials.yaml`、cache、原始 outputs、大型本地结果、`research_ai_supply_chain/`、SPY 旁路改动。
- 每次有效代码/测试/文档进展后，先运行 `git diff --check` 和相关 `pytest`，通过后明确路径 `git add`、提交、推送 `origin/main`。
- heartbeat 已加入 GitHub 同步规则：如果后续自动化产生可提交的 NVDA 进展，必须测试后推送；如果没有可提交 NVDA 变更，则保持静默。

验收标准：
- GitHub 上始终保留最新 NVDA 企业级模型代码和整改记录。
- 不泄露 API 密钥或本地数据。
- 不因自动同步上传无关研究目录。

当前状态：已启用。当前 `origin/main` 最新提交为 `f25283b`，本地暂无未提交 NVDA 代码变更。

## P9：校准与期权诚实度

问题：方向准确率和收益指标不能单独证明模型“准”。如果概率没有校准，`prob_up=0.65` 可能只是一个漂亮数字。期权 snapshot 也只能给终端区间和 skew 线索，不能诚实地产生 path-dependent barrier/drawdown 概率。

解决方案：
- 新增 `calibration` 模块，输出 active signal 的 Brier score、平均校准误差、分箱校准误差。
- 候选晋级的 24/36/60 跨窗口报告纳入校准字段，避免只看收益/胜率。
- 增加 engine rolling calibration，用真实 NVDA 历史窗口直接调用 historical bootstrap、GBM normal、GBM Student-t 三个仿真引擎，检查 PIT 均值、80%/95% 区间覆盖率，为后续动态波动率和校准加权打基础。
- 期权 overlay 明确标注 `path_dependent_metrics_status=unavailable`，不再暗示能从 snapshot 直接得出 barrier/drawdown。

验收标准：
- 每个晋级候选都能看到概率校准指标。
- 后续若引入期权分布/RND，必须先通过覆盖率/尾部校准检验。
- 对无法校准的期权 path 指标，宁可输出不可用，也不能输出伪精确数字。

当前状态：第四版已完成。校准尺子已经从正态代理改为真实仿真引擎，且 realized return 与 engine terminal_returns 统一为 simple return 口径。NVDA 10 年、21 日 horizon、5 年训练窗、`n_sims=5000` 的真实引擎校准显示：`step=5` 下每个方法有 247 个窗口，historical bootstrap 的 80%/95% 覆盖率为 78.14%/91.90%，GBM normal 为 81.78%/92.31%，GBM Student-t 为 80.57%/93.12%；三者 PIT 均值均接近 0.5，PIT KS p-value 分别约 0.86/0.12/0.12。下一步应基于各引擎真实覆盖率选择尾部修正和 ensemble 权重，而不是用代理分布下结论。

## P10：生产 baseline 参数复验

问题：当前 baseline 在 24M/36M 表现稳健，但 60M 年化收益率约 14.67%，低于 15% 硬门槛一点点。直接加高样本规则会提高交易数，但会拖低 60M 胜率和 Sharpe，不能为了 sample size 牺牲生产质量。

已测试方案：
- Risk block：按 downtrend/high-vol/sector/macro/peer stress 投票屏蔽交易。结论是不采纳，收益下降大于风控收益。
- High-sample add-on：baseline 空仓时加入高样本候选。结论是不采纳，交易数上升但胜率和 Sharpe 被拖累。
- Dynamic risk budget：小幅改善，但不能单独解决 60M 年化门槛。
- Stop/take retest：保留 2.5% stop-loss，把 take-profit 从 4.5% 收紧到 4.0%。结论是采纳。

采纳结果：
- 24M：年化 27.10%，Sharpe 2.41，最大回撤 -2.94%，胜率 75.00%，利润因子 6.64，交易数 32。
- 36M：年化 18.71%，Sharpe 1.59，最大回撤 -13.32%，胜率 62.96%，利润因子 3.17，交易数 54。
- 60M：年化 15.40%，Sharpe 1.24，最大回撤 -13.32%，胜率 56.99%，利润因子 2.32，交易数 93。

当前状态：第五版已完成。`production_model.py` 将这个复验过的 stop/take 作为显式 audited override，`rule_from_row()` 在把 optimizer row 转成可执行规则时自动应用。下一步继续优化时，必须用这个 production override 作为新的 baseline，而不是旧的 4.5% take-profit。

## P11：过拟合与脆弱性审计

问题：模型刚刚越过硬门槛，不代表已经稳健。尤其 60M 年化收益率只比 15% 门槛高约 0.40pct，且 stop/take 参数邻域里只有一个组合完整通过，存在参数脆弱性和多重搜索选择偏差。

解决方案：
- 新增 `overfit_audit` 模块，晋级前统一检查：24/36/60 硬门槛边际、train/validation 退化、stop/take 邻域通过数量、年份/状态弱点、概率校准误差、真实引擎区间覆盖。
- 审计输出 `PASS_BUT_FRAGILE` / `PASS_WITH_WARNINGS` / `REJECT_HARD_GATE_FAILURE`，不允许只看一个漂亮回测表。
- 明确控制项：新规则必须通过 24/36/60 复验；2022-like 高波动/弱趋势 regime 独立建模；概率校准未改善前不把 `prob_up` 用作仓位杠杆；未来 stop/take 改动必须有参数邻域证据。

当前状态：第一版已完成。当前 production baseline 硬门槛通过，但审计状态是 `PASS_BUT_FRAGILE`，原因包括：60M 年化安全垫偏薄、参数邻域只有 1/9 通过、短窗口交易数不足 60、2022 年 precision 低于 50%、概率校准弱、95% 区间仍 under-covered。结论是可以继续作为生产 baseline，但不能升仓，也不能停止长跑优化。

## P12：60M / Rolling OOS 稳定晋级整改

问题：此前 high-sample 与 strict optimizer 都主要在 24M 快筛，60M 和 OOS 更像事后报告。结果是候选能在短窗口漂亮，但 60M 复验或最后 45 天 OOS 崩掉。

解决方案：
- 新增 `stable_candidate_optimizer`，从 strict/high-sample repaired 结果中抽取候选，再统一跑 24/36/60M、rolling OOS 和 regime diagnostics。
- `candidate_promotion` 的 60M 逻辑从 “not broken” 改成硬门槛：年化、Sharpe、回撤、胜率、利润因子、交易数和 active days 必须同时达标。
- rolling OOS 检查 45/63/84/126 trading-day 段；OOS 样本不足或 Sharpe degradation 超过 20% 都不能晋级。
- 当前 baseline 没有被替换；新候选只有 `promote_candidate` 才允许进入 production 讨论。

当前状态：第一版已完成。smoke run 显示当前 high-sample 候选 `tw210_sw42_5d_return...` 被稳定验证器判为 `reject_long_window_failure`：24M/36M 强，但 60M 年化、Sharpe、回撤、胜率和交易数不达标；production baseline 继续保留。
