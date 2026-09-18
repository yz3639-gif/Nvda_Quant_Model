# Technical discussion brief

## 30-second project description

I built an auditable NVDA research workflow that compares rule and machine-learning signals under explicit trading assumptions. It uses training-only preprocessing, label-maturity checks, a reconciled fill ledger, and rolling probability and distribution diagnostics. A key finding was that apparently strong historical results were sensitive to execution timing and stop definitions, so the project reports those assumptions and unsuccessful experiments alongside the results.

## Questions worth being able to answer

**Why did the old trade ledger disagree with the account?** Entry equity was captured after the entry cost, so the trade's reported P&L omitted that cost. The account could lose money while the trade appeared to win. The replacement assigns fees to the same-direction trading episode and reconciles realized plus unrealized P&L to cash and shares.

**What happens when a stock gaps through a stop?** With yesterday's close 100, a 2.5% stop and the next open at 80, the model cannot assume a fill at97.5 when the bar's high is85. The explicit gap policy fills at80. With daily bars, an intrabar stop/take conflict still requires a stated convention.

**Why does the historical replay not prove a profitable new model?** The signals were previously selected using historical data. Replaying them can isolate accounting and execution sensitivity, but it does not erase selection bias or reproduce the original fitting history.

**What does point-in-time mean here?** Features and labels have separate availability clocks. The entire label must mature before a training row can be used. A close(t) signal trades at open(t+1), and its target matures at open(t+2). The pipeline tests this by changing later prices and checking that earlier actual model fits and predictions remain unchanged.

**Why not simply select the best Sharpe?** The project had already searched a large historical candidate space. Another attractive historical score can be a selection artifact. The new design preregisters a limited candidate set, scores probabilities and net trading outcomes, preserves failures, and requires prospective evidence for promotion.

**Why keep Normal and Student-t distributions when bootstrap exists?** Their assumptions differ. The comparison tests terminal coverage, tail misses, interval width, PIT and interval score on the same forecast windows. Better coverage without better sharpness is not necessarily improvement.

**What did the news experiment establish?** The saved data are insufficient for a defensible incremental-alpha claim. The infrastructure now preserves availability and story clusters, but empirical validation remains data-limited. Identifying that limit is part of the research result.

## Short LinkedIn description

Python research platform for NVDA with point-in-time validation, reconciled backtesting, probability calibration, and distribution diagnostics. Includes reproducible experiments, execution-sensitivity analysis, and transparent negative results.
