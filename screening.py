from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import os
import sys
import subprocess

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass


# ---------------------------------------------------------------------------
# Credential + client helpers
# ---------------------------------------------------------------------------

def _load_api_keys() -> tuple[str, str]:
    load_dotenv()

    api_key = (
        os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("APCA_API_KEY_ID")
    )
    secret_key = (
        os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
    )

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca credentials missing. Expected ALPACA_API_KEY_ID and "
            "ALPACA_API_SECRET_KEY (or APCA_* variants) in .env."
        )

    return api_key, secret_key


def get_stock_client() -> StockHistoricalDataClient:
    api_key, secret_key = _load_api_keys()
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_trading_client() -> TradingClient:
    api_key, secret_key = _load_api_keys()
    return TradingClient(api_key, secret_key, paper=True)


# ---------------------------------------------------------------------------
# Universe (all tradable symbols)
# ---------------------------------------------------------------------------

def load_all_tradable_symbols() -> List[str]:
    """
    Load a broad universe of symbols from Alpaca.

    - Only active, tradable US equities
    """
    trading_client = get_trading_client()
    assets = trading_client.get_all_assets()

    symbols: List[str] = []
    for asset in assets:
        if not getattr(asset, "tradable", False):
            continue
        status = getattr(asset, "status", None)
        if status != "active":
            continue

        asset_class_val = getattr(asset, "asset_class", None)
        if (
            asset_class_val == AssetClass.US_EQUITY
            or str(asset_class_val).lower() in ("us_equity", "us equity")
        ):
            symbols.append(asset.symbol)

    symbols.sort()
    return symbols


# ---------------------------------------------------------------------------
# Phase 2: historical daily bars (used only for survivors)
# ---------------------------------------------------------------------------

def fetch_recent_daily_bars(
    symbol: str,
    lookback_days: int = 20,
    client: Optional[StockHistoricalDataClient] = None,
) -> pd.DataFrame:
    """
    Fetch recent daily bars for a single symbol.

    Returns a DataFrame indexed by timestamp, sorted ascending.
    """
    if client is None:
        client = get_stock_client()

    end_ts = pd.Timestamp.now(tz="America/New_York")
    start_ts = end_ts - pd.Timedelta(days=lookback_days + 5)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_ts,
        end=end_ts,
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(request)
    if bars is None or bars.df is None or bars.df.empty:
        raise ValueError(f"No daily bar data returned for symbol {symbol!r}")

    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level=0)
        except KeyError:
            raise ValueError(f"No data for {symbol!r} in bar response")

    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Ross Cameron 5 pillars evaluation
# ---------------------------------------------------------------------------

@dataclass
class PillarCheckResult:
    symbol: str
    latest_price: float
    prev_close: float
    today_volume: int
    avg_daily_volume: float
    relative_volume: float

    price_ok: bool
    roc_ok: bool
    rvol_ok: bool
    news_ok: bool
    float_ok: bool

    extra_info: Dict[str, Any]

    @property
    def all_ok(self) -> bool:
        return (
            self.price_ok
            and self.roc_ok
            and self.rvol_ok
            and self.news_ok
            and self.float_ok
        )


def compute_relative_volume(
    today_volume: float,
    recent_volumes: List[float],
) -> float:
    if not recent_volumes:
        return 0.0
    avg = sum(recent_volumes) / len(recent_volumes)
    if avg <= 0:
        return 0.0
    return today_volume / avg


def evaluate_symbol_against_pillars(
    symbol: str,
    latest_price: float,
    prev_close: float,
    today_volume: float,
    recent_daily_volumes: List[float],
    news_items: Optional[List[str]] = None,
    float_shares: Optional[float] = None,
    require_news: bool = False,
    require_float: bool = False,
) -> PillarCheckResult:
    """
    Core Ross 5 pillars check.
    """

    price_ok = 2.0 <= latest_price <= 20.0

    roc = 0.0
    if prev_close > 0:
        roc = (latest_price - prev_close) / prev_close
    roc_ok = roc >= 0.10

    rvol = compute_relative_volume(today_volume, recent_daily_volumes)
    rvol_ok = rvol >= 5.0

    if require_news:
        news_ok = bool(news_items)
    else:
        news_ok = True

    if require_float:
        float_ok = float_shares is not None and float_shares < 20_000_000
    else:
        float_ok = True

    avg_vol = (
        sum(recent_daily_volumes) / len(recent_daily_volumes)
        if recent_daily_volumes
        else 0.0
    )

    return PillarCheckResult(
        symbol=symbol,
        latest_price=latest_price,
        prev_close=prev_close,
        today_volume=int(today_volume),
        avg_daily_volume=avg_vol,
        relative_volume=rvol,
        price_ok=price_ok,
        roc_ok=roc_ok,
        rvol_ok=rvol_ok,
        news_ok=news_ok,
        float_ok=float_ok,
        extra_info={
            "roc_pct": roc * 100.0,
            "num_recent_days": len(recent_daily_volumes),
            "news_count": len(news_items) if news_items else 0,
            "float_shares": float_shares,
        },
    )


def evaluate_symbol_with_recent_data(
    symbol: str,
    lookback_days: int = 20,
    client: Optional[StockHistoricalDataClient] = None,
    require_news: bool = False,
    require_float: bool = False,
) -> Optional[PillarCheckResult]:
    """
    Fetch recent daily bars for a symbol and run Ross's pillars on it.
    """
    if client is None:
        client = get_stock_client()

    try:
        df = fetch_recent_daily_bars(symbol, lookback_days=lookback_days, client=client)
    except Exception as e:
        print(f"[WARN] Failed to fetch bars for {symbol}: {e}")
        return None

    if len(df) < 2:
        print(f"[WARN] Not enough data for {symbol} (need at least 2 daily bars)")
        return None

    latest_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]

    latest_price = float(latest_bar["close"])
    prev_close = float(prev_bar["close"])
    today_volume = float(latest_bar["volume"])

    recent_hist = df.iloc[:-1]
    recent_hist = recent_hist.tail(lookback_days)
    recent_volumes = [float(v) for v in recent_hist["volume"].tolist()]

    news_items: List[str] = []
    float_shares: Optional[float] = None

    return evaluate_symbol_against_pillars(
        symbol=symbol,
        latest_price=latest_price,
        prev_close=prev_close,
        today_volume=today_volume,
        recent_daily_volumes=recent_volumes,
        news_items=news_items,
        float_shares=float_shares,
        require_news=require_news,
        require_float=require_float,
    )


# ---------------------------------------------------------------------------
# Phase 1: coarse filter using batched daily bars
# ---------------------------------------------------------------------------

def phase1_filter_candidates(
    all_symbols: List[str],
    min_price: float,
    max_price: float,
    min_today_volume: int,
    client: Optional[StockHistoricalDataClient] = None,
    batch_size: int = 200,
) -> List[str]:
    """
    Phase 1: Use daily bars in batches to cheaply filter the universe by:

    - price in [min_price, max_price]
    - today's volume >= min_today_volume
    """
    if client is None:
        client = get_stock_client()

    total = len(all_symbols)
    print(
        f"[Phase 1] Coarse filtering {total} symbols "
        f"for price in [{min_price}, {max_price}] "
        f"and today's volume >= {min_today_volume:,}..."
    )

    candidates: List[str] = []

    end_ts = pd.Timestamp.now(tz="America/New_York")
    start_ts = end_ts - pd.Timedelta(days=5)

    for start in range(0, total, batch_size):
        batch = all_symbols[start:start + batch_size]
        if not batch:
            continue

        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start_ts,
            end=end_ts,
            feed=DataFeed.IEX,
        )

        try:
            bars = client.get_stock_bars(request)
        except Exception as e:
            print(f"[WARN] Bar request failed for batch starting at {start}: {e}")
            continue

        if bars is None or bars.df is None or bars.df.empty:
            print(f"[WARN] No bars returned for batch starting at {start}")
            continue

        df = bars.df

        if not isinstance(df.index, pd.MultiIndex):
            print(f"[WARN] Unexpected bar index type for batch starting at {start}")
            continue

        for sym in batch:
            try:
                sym_df = df.xs(sym, level=0)
            except KeyError:
                continue

            if sym_df.empty:
                continue

            sym_df = sym_df.sort_index()
            latest_bar = sym_df.iloc[-1]

            try:
                latest_price = float(latest_bar["close"])
                today_volume = float(latest_bar["volume"] or 0)
            except Exception:
                continue

            if not (min_price <= latest_price <= max_price):
                continue
            if today_volume < min_today_volume:
                continue

            candidates.append(sym)

        print(
            f"[Phase 1] Processed {min(start + batch_size, total)}/{total} symbols. "
            f"Current candidates: {len(candidates)}"
        )

    print(f"[Phase 1] Finished. {len(candidates)} symbols passed coarse filters.\n")
    return candidates


# ---------------------------------------------------------------------------
# Backtest invocation helpers (subprocesses)
# ---------------------------------------------------------------------------

def ensure_daily_data_for_symbol(symbol: str, auto_download: bool) -> bool:
    """
    Ensure data/{symbol}_daily.csv exists.

    If auto_download is True and the file is missing, call
    tools/download_history.py --symbol SYMBOL to create it.

    Returns True if the file exists after this function, False otherwise.
    """
    csv_path = os.path.join("data", f"{symbol}_daily.csv")
    if os.path.exists(csv_path):
        return True

    if not auto_download:
        print(
            f"[Data] Missing {csv_path}. "
            f"Run tools/download_history.py --symbol {symbol} "
            f"(or use --auto-download)."
        )
        return False

    print(f"[Data] {csv_path} missing. Auto-downloading DAILY history for {symbol}...")
    cmd = [sys.executable, "tools/download_history.py", "--symbol", symbol]
    try:
        subprocess.run(cmd, check=False)
    except OSError as e:
        print(f"[Data] Failed to run download_history.py for {symbol}: {e}")
        return False

    if os.path.exists(csv_path):
        print(f"[Data] Downloaded {csv_path}.")
        return True
    else:
        print(
            f"[Data] After auto-download, {csv_path} still not found. "
            f"Skipping {symbol}."
        )
        return False


def ensure_intraday_data_for_symbol(symbol: str, auto_download: bool) -> bool:
    """
    Ensure data/{symbol}_intraday.csv exists.

    If auto_download is True and the file is missing, call
    tools/download_intraday_history.py --symbol SYMBOL to create it.

    Returns True if the file exists after this function, False otherwise.
    """
    csv_path = os.path.join("data", f"{symbol}_intraday.csv")
    if os.path.exists(csv_path):
        return True

    if not auto_download:
        print(
            f"[Data] Missing {csv_path}. "
            f"Run tools/download_intraday_history.py --symbol {symbol} "
            f"(or use --auto-download)."
        )
        return False

    print(f"[Data] {csv_path} missing. Auto-downloading INTRADAY history for {symbol}...")
    cmd = [sys.executable, "tools/download_intraday_history.py", "--symbol", symbol]
    try:
        subprocess.run(cmd, check=False)
    except OSError as e:
        print(f"[Data] Failed to run download_intraday_history.py for {symbol}: {e}")
        return False

    if os.path.exists(csv_path):
        print(f"[Data] Downloaded {csv_path}.")
        return True
    else:
        print(
            f"[Data] After auto-download, {csv_path} still not found. "
            f"Skipping {symbol}."
        )
        return False


def run_daily_backtest_for_symbol(
    symbol: str,
    signal_mode: Optional[str] = None,
    auto_download: bool = False,
) -> None:
    """
    Call backtest.py in a subprocess for a single symbol,
    ensuring daily data exists (and optionally auto-downloading it).
    """
    if not ensure_daily_data_for_symbol(symbol, auto_download=auto_download):
        print(f"[Backtest-Daily] Skipping {symbol}: no daily data.")
        return

    cmd = [sys.executable, "backtest.py", "--symbol", symbol]
    if signal_mode:
        cmd.extend(["--signal", signal_mode])

    print(f"[Backtest-Daily] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except OSError as e:
        print(f"[Backtest-Daily] Failed to run backtest.py for {symbol}: {e}")

def save_live_watchlist(symbols: List[str]) -> None:
    """
    Save a simple newline-delimited list of ticker symbols for live trading.
    Each line in data/live_watchlist.txt will be something like:
      TSLA
      GME
      PMI
    """
    with open(LIVE_WATCHLIST_PATH, "w") as file_handle:
        for symbol in symbols:
            symbol_str = symbol.strip()
            if not symbol_str:
                continue
            file_handle.write(symbol_str + "\n")
    print(f"✓ Wrote {len(symbols)} symbols to {LIVE_WATCHLIST_PATH}")

def run_intraday_backtest_for_symbol(
    symbol: str,
    auto_download: bool = False,
) -> None:
    """
    Call backtest_intraday.py in a subprocess for a single symbol,
    ensuring both daily and intraday data exist (and optionally auto-downloading).
    """

    # Intraday backtest needs both DAILY and INTRADAY history
    if not ensure_daily_data_for_symbol(symbol, auto_download=auto_download):
        print(f"[Backtest-Intraday] Skipping {symbol}: no daily data.")
        return

    if not ensure_intraday_data_for_symbol(symbol, auto_download=auto_download):
        print(f"[Backtest-Intraday] Skipping {symbol}: no intraday data.")
        return

    cmd = [sys.executable, "backtest_intraday.py", "--symbol", symbol]

    print(f"[Backtest-Intraday] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except OSError as e:
        print(f"[Backtest-Intraday] Failed to run backtest_intraday.py for {symbol}: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan symbols using Ross Cameron's 5 pillars (two-phase scanner)."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to scan, e.g. 'TSLA,MNDR,AMD'. "
             "If omitted and --scan-all is used, scans all tradable symbols.",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan all tradable US equity symbols from Alpaca.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Number of past trading days to use for average volume (default: 20)",
    )
    parser.add_argument(
        "--require-news",
        action="store_true",
        help="Require that the symbol has news (pillar 4). Currently stubbed.",
    )
    parser.add_argument(
        "--require-float",
        action="store_true",
        help="Require that float < 20M shares (pillar 5). Currently stubbed.",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=2.0,
        help="Minimum price for coarse filter in Phase 1 (default: 2.0)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=20.0,
        help="Maximum price for coarse filter in Phase 1 (default: 20.0)",
    )
    parser.add_argument(
        "--min-today-volume",
        type=int,
        default=50_000,
        help="Minimum today's volume for coarse filter in Phase 1 (default: 50,000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of symbols per daily-bar batch in Phase 1 (default: 200)",
    )

    # Backtest integration flags
    parser.add_argument(
        "--backtest-daily",
        action="store_true",
        help="After screening, run backtest.py on each passing symbol.",
    )
    parser.add_argument(
        "--backtest-signal",
        type=str,
        default=None,
        help="If provided, passed as --signal to backtest.py "
             "(e.g. 'breakout' or 'sma_trend').",
    )
    parser.add_argument(
        "--backtest-intraday",
        action="store_true",
        help="After screening, run backtest_intraday.py on each passing symbol.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="If set, automatically download missing daily/intraday CSVs before backtesting.",
    )

    args = parser.parse_args()

    stock_client = get_stock_client()

    # Determine universe
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"Using explicit symbol list: {len(symbols)} symbols.")
        print("[Phase 1] Skipped (explicit symbols) – running full pillars on provided list.\n")
        candidate_symbols = symbols
    elif args.scan_all:
        print("Loading all tradable symbols from Alpaca...")
        universe = load_all_tradable_symbols()
        print(f"Total tradable US equity symbols: {len(universe)}\n")

        candidate_symbols = phase1_filter_candidates(
            all_symbols=universe,
            min_price=args.min_price,
            max_price=args.max_price,
            min_today_volume=args.min_today_volume,
            client=stock_client,
            batch_size=args.batch_size,
        )
    else:
        raise SystemExit(
            "You must either provide --symbols or use --scan-all to scan all tradable symbols."
        )

    # Phase 2
    print(
        f"[Phase 2] Evaluating {len(candidate_symbols)} candidate symbols "
        f"with lookback_days={args.lookback_days}...\n"
    )

    passing: List[PillarCheckResult] = []

    for sym in candidate_symbols:
        result = evaluate_symbol_with_recent_data(
            symbol=sym,
            lookback_days=args.lookback_days,
            client=stock_client,
            require_news=args.require_news,
            require_float=args.require_float,
        )
        if result is None:
            continue

        status = "✅ PASSES" if result.all_ok else "❌ FAILS"

        print(f"{sym}: {status}")
        print(
            f"  Price:       ${result.latest_price:.2f}  "
            f"(prev close: ${result.prev_close:.2f})"
        )
        print(
            f"  ROC today:   {result.extra_info['roc_pct']:.2f}%  "
            f"-> {'OK' if result.roc_ok else 'X'}"
        )
        print(f"  Volume:      {result.today_volume:,.0f}")
        print(f"  Avg volume:  {result.avg_daily_volume:,.0f}")
        print(
            f"  Rel volume:  {result.relative_volume:.2f}x "
            f"-> {'OK' if result.rvol_ok else 'X'}"
        )
        print(f"  Price OK:    {result.price_ok}")
        print(f"  News OK:     {result.news_ok}  (require_news={args.require_news})")
        print(f"  Float OK:    {result.float_ok} (require_float={args.require_float})")
        print()

        if result.all_ok:
            passing.append(result)

    print("------------------------------------------------------")
    if passing:
        print("Symbols passing all pillars:")
        for r in passing:
            print(
                f"  - {r.symbol} "
                f"(rel vol {r.relative_volume:.2f}x, "
                f"ROC {r.extra_info['roc_pct']:.2f}%)"
            )

        # NEW: write just the symbols once
        passing_symbols = [r.symbol for r in passing]
        save_live_watchlist(passing_symbols)
    else:
        print("No symbols passed all pillars with the current settings.")

    # --------------------------------------------------
    # Optional: run backtests on passing symbols
    # --------------------------------------------------
    if passing and (args.backtest_daily or args.backtest_intraday):
        print("\n[Backtest] Running backtests for passing symbols...")

        for r in passing:
            sym = r.symbol

            if args.backtest_daily:
                run_daily_backtest_for_symbol(
                    symbol=sym,
                    signal_mode=args.backtest_signal,
                    auto_download=args.auto_download,
                )

            if args.backtest_intraday:
                run_intraday_backtest_for_symbol(
                    symbol=sym,
                    auto_download=args.auto_download,
                )


if __name__ == "__main__":
    main()
