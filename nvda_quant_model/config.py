from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class StrategyConfig:
    ticker: str = "NVDA"
    benchmark: str = "^GSPC"
    start_date: str = "auto"
    end_date: str = "latest"
    lookback_months: int = 24
    initial_capital: float = 100_000.0
    position_unit: float = 0.10
    max_exposure: float = 1.00
    commission: float = 0.001
    slippage: float = 0.0005
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.05
    max_drawdown_limit: float = 0.20
    train_window: int = 189
    test_window: int = 42
    walk_forward_jobs: int = 1
    top_k_features: int = 12
    include_fundamentals: bool = False
    include_peer_events: bool = True
    signal_threshold: float = 0.55
    min_expected_return: float = 0.0005
    rule_quantile: float = 0.55
    rule_max_filters: int = 2
    fast_rule_only: bool = False
    model_params_path: str | None = None
    precision_rule_path: str | None = None
    random_state: int = 42

    @property
    def round_trip_cost(self) -> float:
        return self.commission + self.slippage


MACRO_TICKERS = {
    "VIX": "^VIX",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "USD": "DX-Y.NYB",
    "USD_FALLBACK": "UUP",
    "TREASURY_10Y": "^TNX",
    "HYG": "HYG",
    "LQD": "LQD",
}


SECTOR_TICKERS = {
    "XLK": "XLK",
    "SMH": "SMH",
    "QQQ": "QQQ",
}


PEER_TICKERS = {
    "AMD": "AMD",
    "AVGO": "AVGO",
    "TSM": "TSM",
    "ASML": "ASML",
    "MU": "MU",
    "QCOM": "QCOM",
    "INTC": "INTC",
    "ARM": "ARM",
}


RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
