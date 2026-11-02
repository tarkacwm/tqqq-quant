"""Ticker-aware research data adapters. No price fetching inside the strategy engine."""
from __future__ import annotations
import re
import pandas as pd
from .core import load_prices

TICKER_RE = re.compile(r'^[A-Z0-9^][A-Z0-9.^=\-]{0,19}$')

def normalize_ticker(ticker: str) -> str:
    symbol = str(ticker).strip().upper()
    if not TICKER_RE.fullmatch(symbol):
        raise ValueError('Enter a valid ticker (letters, digits, ^, ., -, or =).')
    return symbol

def csv_tickers(upload) -> list[str]:
    upload.seek(0)
    raw = pd.read_csv(upload, nrows=1000)
    upload.seek(0)
    symbol_col = next((c for c in raw.columns if c.lower() in ('ticker', 'symbol')), None)
    if symbol_col:
        return sorted({normalize_ticker(s) for s in raw[symbol_col].dropna().unique()})
    return []

def load_ticker_csv(upload, ticker: str | None = None) -> pd.DataFrame:
    upload.seek(0)
    raw = pd.read_csv(upload)
    upload.seek(0)
    symbol_col = next((c for c in raw.columns if c.lower() in ('ticker', 'symbol')), None)
    if symbol_col:
        if not ticker:
            raise ValueError('Select a ticker from this multi-symbol CSV.')
        chosen = normalize_ticker(ticker)
        raw = raw.loc[raw[symbol_col].astype(str).str.upper().str.strip() == chosen].copy()
        if raw.empty:
            raise ValueError(f'No records found for {chosen}.')
    # Support generic YCharts price exports and common vendor column conventions.
    if 'Adj Close' in raw and 'Close Price' not in raw and 'TQQQ Close' not in raw:
        raw = raw.rename(columns={'Adj Close': 'Close Price'})
    elif 'close' not in raw and 'Close Price' not in raw and 'TQQQ Close' not in raw and 'Close' not in raw:
        raise ValueError('CSV requires a Close/Close Price/Adj Close column.')
    return load_prices(raw)

def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily bars; end is exclusive per yfinance convention."""
    import yfinance as yf
    symbol = normalize_ticker(ticker)
    bars = yf.download(symbol, start=start, end=end, auto_adjust=True,
                       progress=False, threads=False)
    if bars.empty:
        raise ValueError(f'No price data returned for {symbol} in the selected period.')
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    bars = bars.reset_index()
    bars = bars.rename(columns={'Close': 'Close Price', 'Datetime': 'Date'})
    return load_prices(bars)
