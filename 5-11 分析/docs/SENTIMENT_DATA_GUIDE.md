# Sentiment Data 手动填写指南

## 文件位置
`manual_sentiment.yaml` 在项目根目录

## 1. AAII Bull-Bear Spread
**去哪**: https://www.aaii.com/sentimentsurvey
**找什么**:
- 主表第一行 "Bullish" 和 "Bearish" 的百分比
- 计算: Bull% - Bear% = spread

**例子**:
- 看到 Bullish: 35.5%, Bearish: 28.2%
- 填: AAII_bull_bear_spread: 7.3
- AAII_date 用表头的 "Week ended" 日期

**频率**: 每周四更新

## 2. CFTC COT
**去哪**: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
**找什么**:
- 点 "Current Legacy Report" → "Short Format"
- Ctrl+F 搜 "E-MINI S&P 500" (大写)
- 在 Non-Commercial (投机者) 行: Net = Long - Short

**E-MINI S&P 500 例子**:
- Long: 450,000; Short: 270,000
- Net: +180,000
- 填: CFTC_emini_spx_net: 180000

**10Y T-NOTE**: Ctrl+F 搜 "10-YEAR U.S. TREASURY"

**分位数**:
- 笨办法: 拿过去 260 周 Net 数列排序看位置
- 偷懒办法: https://www.barchart.com/futures/commitment-of-traders/percent-of-open-interest/legacy-futures/ES 看图

**频率**: 每周五 15:30 ET, 数据截止上周二

## 3. CBOE Put/Call Ratio
**去哪**: https://www.cboe.com/us/options/market_statistics/daily/
**偷懒办法**: https://ycharts.com/indicators/cboe_equity_put_call_ratio 看图目测 10 日均值
**频率**: 每日

## 4. NAAIM Exposure Index
**去哪**: https://www.naaim.org/programs/naaim-exposure-index/
**找什么**: 页面顶部大字数字, 例: "NAAIM Number: 78.45"
**参考线**:
- < 30 = 极度防御
- 30-60 = 中性偏空
- 60-90 = 中性偏多
- > 90 = 极度激进
**频率**: 每周三晚

## 5. FINRA Margin Debt
**去哪**: https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics
**找什么**: "Debit Balances in Customers' Securities Margin Accounts" 最新月 vs 12 个月前
**例子**:
- 4 月 850B vs 去年 4 月 745B
- YoY = (850-745)/745 = 0.1409
- 填: FINRA_margin_debt_yoy: 0.14
**频率**: 月度

## 6. ICI Fund Flows
**去哪**: https://www.ici.org/research/stats/flows
**找什么**: "Weekly Estimated Long-Term Mutual Fund Flows" 最近 4 周 Domestic + World Equity 合计
**频率**: 每周三

## 5 分钟工作流
1. ⏱️ 1 分钟 — AAII
2. ⏱️ 3 分钟 — CFTC COT
3. ⏱️ 1 分钟 — CBOE P/C
4. ⏱️ 1 分钟 — NAAIM
5. ⏱️ 1 分钟 — FINRA (月度)
6. ⏱️ 1 分钟 — ICI
7. ⏱️ 2 分钟 — 检查 + 跑模型

## 替代数据源
- SentimenTrader ($50/月) — 全部汇总
- TradingView Sentiment — 看图目测
- StockCharts — 免费 AAII/NAAIM 历史图
