# NVDA research: technical discussion brief

This project studies daily NVDA equity signals, probability calibration and trading assumptions. Its strongest evidence is a reproducible research process: explicit decision clocks, independently reconcilable P&L, bounded model comparisons and visible failure cases. It does not establish a live trading edge or price NVDA options.

## The project in 30 seconds

I built a Python workflow comparing a volume/momentum rule, logistic regression and shallow boosting across 18 chronological outer folds. It separates forecasting quality from position sizing and transaction costs, preserves trade-level accounting, and evaluates predictive distributions as well as strategy returns. Replaying the same historical signals exposed material sensitivity to execution timing and stop definitions. The results include unsuccessful strategies, so a reviewer can see what the evidence supports and what still needs validation.

## What was actually evaluated

| Component | Implemented design | Evidence |
|---|---|---|
| Data | 1,641 daily OHLCV rows, 2019-11-11 through 2026-05-22 | [Frozen manifest](results/frozen_20260918/manifest.json) |
| Directional prediction | Three model families; five registered settings; 18 outer folds; 1,075 predictions per family | [Folds](results/frozen_20260918/folds.json), [predictions](results/frozen_20260918/predictions.csv) |
| Calibration | Raw, training-base-rate shrinkage and sigmoid mappings; selection within chronological training blocks | [Probability scores](results/frozen_20260918/probability_scores.csv) |
| Trading | 17 strategy/control variants, costs at 1×/2×/4× and 36 nearby stop/take settings | [Strategy results](results/frozen_20260918/strategy_summary.csv), [cost tests](results/frozen_20260918/cost_stress.csv) |
| Distributions | Bootstrap, Normal and Student-t models, each with rolling/EWMA volatility: six variants at 5- and 21-session horizons | [Distribution results](results/frozen_20260918/distribution_summary.csv) |
| Execution audit | 12 frozen-signal replay cases separating legacy, corrected accounting and changed trading rules | [Replay comparison](results/frozen_20260918/historical_replay/historical_replay.csv) |

The **11-candidate budget** is five directional settings plus six distribution variants, not 11 independent directional model families. Strategy controls and stress cases are comparisons built around those registered candidates; they are not independent replications.

The saved experiment was generated on **2026-09-18** from prices ending **2026-05-22**. A separate historical default-strategy replay ends **2026-07-24**. Neither a new documentation date nor publishing this repository refreshes those inputs. Reused historical data remain development evidence.

## Questions to discuss with a trader or researcher

**What does this demonstrate that transfers to a trading desk?** The workflow makes signal, execution and risk assumptions inspectable. Each fill contributes to the same account identity; each forecast has a defined information clock; and comparisons retain costs, exposure and adverse cases. Those are useful foundations for desk analytics. This NVDA study does not demonstrate commodity-option calibration, exchange connectivity or production market making.

**Why did the old trade ledger disagree with the account?** Entry equity was captured after the entry cost, so the trade's reported P&L omitted that cost. The account could lose money while the trade appeared to win. The replacement assigns fees to the same-direction trading episode and verifies both `equity = cash + shares × mark` and `equity = initial capital + realized P&L + unrealized P&L`.

**What happens when the market gaps through a stop?** Suppose yesterday's close was 100, the stop is 97.50 and the next bar opens at 80 with a high of 85. A fill at 97.50 would be impossible on that bar; the explicit gap policy fills at 80. If both stop and take-profit levels are touched within a daily bar, the default convention is stop-first because the bar does not reveal their ordering.

**What did replay establish?** In the 60-month case, annualized return was 15.40% under the legacy close/daily-reset rules, 15.80% under corrected close/daily-reset accounting, and −0.65% under next-open/entry-reference rules. The last comparison changes the trading definition as well as execution timing. It is not a pure bug-fix effect and does not establish a new profitable model. The familiar legacy Sharpe of 1.24 belongs to the original historical case, not to every v2 model or a newly untouched test set.

**How is future information excluded within a fold?** Features and labels have separate availability clocks. A close(t) decision predicts open(t+1) to open(t+2), and a training label must have matured before fitting. Preprocessing is fitted only on eligible training rows. Tests mutate later prices and check that earlier actual model fits and predictions remain unchanged. These controls address implementation leakage; they cannot erase earlier research on the same historical period.

**Does a better directional forecast imply better trading P&L?** No. The forecast target is an open-to-open return, while stop/take exits, rebalancing and drawdown halts can change the realized holding period. Costs and exposure also change the economic outcome. Forecast scores and strategy ledgers are reported separately, and horizon-aligned policy comparisons are a next validation step.

**What do the new model results show?** Selected logistic calibration has a Brier score of approximately 0.24957, compared with 0.25322 raw and 0.24965 for the training base rate. The descriptive block interval for improvement over the base rate crosses zero. The logistic volatility-targeted strategy lost 6.98% net in the saved simulated account. Positive low-turnover exposure controls do not establish directional alpha; some have very few completed trades. No model is automatically promoted.

**Why compare Normal, Student-t and bootstrap distributions?** Each makes different assumptions about tails and dependence. The same forecast windows are scored for terminal coverage, tail misses, interval width, PIT and interval score. Wider intervals can improve coverage without producing a more useful forecast. Student-t log shocks are not used to claim a finite expected price or to price options.

**What did the news experiment establish?** There are only 26 saved records over roughly three months, with insufficient historical availability evidence. The experiment stops at its eligibility gate. News can provide context here; it does not support an incremental-alpha claim.

## The next credible evidence

1. Freeze a model and its execution policy before collecting new outcomes; register fresh, time-stamped forecasts using the [shadow workflow](RUNBOOK.md#prospective-shadow-evidence).
2. Align the probability target and trading horizon in a separately registered comparison, preserving stopped and held-to-horizon outcomes.
3. Validate costs and fill assumptions against appropriate market data rather than a daily-bar liquidity proxy.
4. Report forecast quality, net P&L, turnover, exposure and failure regimes together. A fresh download alone does not create a prospective record.

## Concise project description

Built a Python framework comparing three daily NVDA signal models across 18 walk-forward folds, with transaction-cost stress tests at 1×, 2× and 4× baseline costs; added reconciled fill accounting and execution-sensitivity analysis.

Use that description with the dated [model card](MODEL_CARD.md) and [saved research report](results/frozen_20260918/research_report.md). The numerical scope is reproducible; it is not a claim of profitable live performance.
