# SPY 概率分布量化 CLI

这是一个生产级别的 Python 命令行工具，用于估算 SPY 或其他可由 yfinance 获取的标的，从下一个交易日开始、未来一段交易日内的价格变动概率分布。输出是模型假设下的统计分布，不是投资建议，也不是对未来结果的断言。

## 快速开始

```sh
python3 -m pip install -r requirements.txt
python3 spy_forecast.py --ticker SPY --horizon-days 63 --n-sims 10000 --plot
```

输出默认保存到 `./output/`：

- `forecast_results.json`: 完整参数、假设、警告、指标和集成平均
- `probability_metrics.csv`: 每种方法与集成平均的扁平化概率指标
- `method_comparison.csv`: 三种主方法并列表，含 >5 个百分点分歧标记
- `terminal_return_histogram.png`: 三种方法终值收益率直方图叠加
- `price_path_fan_chart.png`: 每种方法价格路径 50/80/95% 置信带
- `terminal_return_cdf.png`: 三种方法 CDF 对比
- `sample_price_paths.png`: 每种方法 100 条样本路径
- `risk_neutral_vs_historical_density.png`: 风险中性密度与历史密度对比

## CLI 参数

```sh
python3 spy_forecast.py [选项]

--ticker TEXT              默认 SPY
--horizon-days INTEGER     默认 63
--start-date DATE          默认下一个工作日
--lookback-years FLOAT     默认 20
--n-sims INTEGER           默认 10000
--block-size INTEGER       默认 5
--risk-free-rate FLOAT     默认自动从 ^IRX 获取
--thresholds TEXT          默认 "-10,-5,-2,0,2,5,10"
--output-dir PATH          默认 ./output
--options-source TEXT      默认 yfinance
--plot / --no-plot         默认 --plot
--seed INTEGER             默认 42
--validate                 从约一年前做一次历史回测验证
```

下载数据会缓存到 `./cache/`。重复运行时，历史价格、无风险利率和已选择的期权链会优先使用本地缓存。

## 三种方法

### 1. Historical Bootstrap

工具拉取过去 `--lookback-years` 年的日度调整收盘价，计算对数收益率，并使用移动块自助法生成前向路径。默认块大小是 5 个交易日，用来保留一部分短期自相关和波动聚集结构。

局限性：它假设历史窗口仍能代表未来状态；如果市场发生 regime shift，历史分布可能失效。

### 2. GBM Monte Carlo

工具使用同一历史窗口估计日度对数收益均值和标准差，并年化为：

- `mu_annual = mean_daily * 252`
- `sigma_annual = std_daily * sqrt(252)`

然后使用 GBM 公式模拟路径。另有 Student-t 版本，使用 `scipy.stats.t.fit` 通过最大似然估计自由度，并用标准化 t 冲击替代正态冲击，以呈现肥尾敏感性。集成平均和三方法对比表使用正态 GBM 作为蒙特卡洛主结果，Student-t 作为补充结果单独报告。

局限性：GBM 假设漂移和波动率恒定，普通版本还假设冲击正态独立。

### 3. Option-Implied Distribution

工具通过 yfinance 拉取 SPY 期权链，选择最接近预测期限的到期日。期限匹配使用 `horizon_days * 365 / 252` 把交易日近似换成日历日。

优先路径：

1. 从平值附近期权提取 ATM IV。
2. 使用看涨期权价格对执行价做有限差分，按 Breeden-Litzenberger 公式估计风险中性密度：

```text
f(K) = exp(rT) * d²C / dK²
```

3. 如果期权链稀疏或噪声过大，明确降级为 ATM IV 参数化的风险中性对数正态分布。

局限性：期权隐含分布是风险中性分布，不等同于真实世界概率；它包含风险溢价、供需、事件风险和流动性影响。

## 输出指标解读

每种方法都会输出：

- 上涨概率 `P(up)` 和下跌概率 `P(down)`
- 用户阈值定义的收益分档概率
- 期望收益、中位数收益、终值收益标准差、偏度、超额峰度
- 50%、80%、90%、95%、99% 置信区间，分别给出收益率和终值价格
- 95%、99% VaR 与 CVaR，数值按左尾收益率报告，通常为负数
- 最大回撤分布：均值、中位数和 95% 最坏损失
- 路径相依障碍触及概率：持有期内任意一天触及 `+/-5%`、`+/-10%`、`+/-15%`、`+/-20%`

三方法对比表会并排展示 `P(up)`、期望收益、95% VaR 和 1-sigma 区间。任一列最大差异超过 5 个百分点时，终端表格会用 `*` 标出，并输出可能原因，例如隐含波动率显著高于历史波动率可能表示市场正在为事件风险、对冲需求或波动率风险溢价定价。

## 验证

```sh
python3 spy_forecast.py --validate --no-plot
```

验证模式会从大约一年前选取一个历史 as-of 日期，用当时之前的数据生成 Historical Bootstrap、GBM Normal 和 GBM Student-t 分布，再对比后续真实 `--horizon-days` 的实现收益。免费 yfinance 不提供历史期权链，因此 option-implied 方法的历史验证会被明确跳过。

## 免责声明

本工具只输出特定假设下的统计估计。它不是投资建议，不应被解释为未来会发生的概率真相。所有方法都可能在市场状态切换、流动性改变、宏观冲击、波动率结构变化或数据质量问题下失效。

---

# Existing Alpaca Integration

这个项目已经接入 Alpaca Trading API。默认使用 paper trading，只有显式设置 `ALPACA_ALLOW_LIVE_TRADING=true` 时才允许向 live trading 发出写请求。

## 快速开始

1. 创建本地环境变量文件：

```sh
cp .env.example .env
```

2. 在 `.env` 填入 Alpaca 凭证：

```sh
APCA_API_KEY_ID=your_key_id
APCA_API_SECRET_KEY=your_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

3. 检查连接：

```sh
npm run alpaca:check
```

## 代码使用

```js
import { createAlpacaClient, loadDotEnv } from './src/index.js';

loadDotEnv();

const alpaca = createAlpacaClient();
const account = await alpaca.getAccount();
const clock = await alpaca.getClock();

console.log(account.status, clock.is_open);
```

## 可用方法

- `getAccount()`
- `getClock()`
- `getAsset(symbol)`
- `listPositions()`
- `listOrders(params)`
- `submitOrder(order)`
- `cancelOrder(orderId)`
- `getLatestStockQuote(symbol)`

## 环境变量

- `APCA_API_KEY_ID`: Alpaca API key ID
- `APCA_API_SECRET_KEY`: Alpaca secret key
- `APCA_API_BASE_URL`: 交易 API 地址，默认 `https://paper-api.alpaca.markets`
- `APCA_DATA_BASE_URL`: 行情 API 地址，默认 `https://data.alpaca.markets`
- `ALPACA_ALLOW_LIVE_TRADING`: live trading 写请求开关，默认 `false`

## 参考

- [Alpaca Authentication](https://docs.alpaca.markets/v1.3/reference)
- [Alpaca Create an Order](https://docs.alpaca.markets/v1.3/reference/orders)
- [Alpaca Trading Orders Guide](https://docs.alpaca.markets/docs/trading/orders/)
