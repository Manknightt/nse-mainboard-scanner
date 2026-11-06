#!/usr/bin/env python3
"""
NSE mainboard EOD scanner.
Output schema is intentionally identical to the user's 46-column sample.
Universe: NSE Securities Available for Trading - Equity segment, excluding SME.
Prices/volume history: Yahoo Finance via yfinance (.NS symbols).
"""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("nse-scanner")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")
NSE_HOME = "https://www.nseindia.com/"
NSE_SECURITIES_PAGE = "https://www.nseindia.com/static/market-data/securities-available-for-trading"

# Exact 46-column schema from eod-scan-equity-2026-11-05.csv
COLUMNS = [
    "Symbol", "Last Close", "Last Volume", "Avg Vol",
    "% Chg 1 Day", "% Chg 1 Week", "% Chg 1 Month", "% Chg 3 Months", "% Chg 1 Year",
    "SMA 5", "EMA 5", "SMA 13", "EMA 13", "SMA 20", "EMA 20", "SMA 34", "EMA 34",
    "SMA 50", "EMA 50", "SMA 89", "EMA 89", "SMA 200", "EMA 200",
    "BBand Upper", "BBand Lower", "BBand Mid",
    "MACD", "MACD Signal", "MACD Hist", "ADX", "DMI+", "DMI-", "RSI",
    "Stoch %K", "Stoch %D", "CCI", "PSAR", "True Range", "ATR", "Will %R",
    "TRIX", "StochRSI %K", "StochRSI %D", "Momentum", "Candle", "Date"
]
# Mainboard equity series. SME is published as a separate NSE download and is not used.
MAINBOARD_SERIES = {"EQ", "BE", "BZ", "BT", "ST"}

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/json,*/*",
                      "Referer": NSE_HOME, "Accept-Language": "en-US,en;q=0.9"})
    try:
        s.get(NSE_HOME, timeout=20)
    except requests.RequestException:
        pass
    return s

def fetch_universe() -> pd.DataFrame:
    """Read NSE's official equity-segment CSV link from its securities page."""
    s = get_session()
    page = s.get(NSE_SECURITIES_PAGE, timeout=30)
    page.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page.text, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).lower().split())
        href = a["href"]
        if "equity segment" in label and ".csv" in label:
            candidates.append(requests.compat.urljoin(NSE_SECURITIES_PAGE, href))
    # Fallback: inspect links whose accessible label mentions the equity CSV.
    if not candidates:
        for a in soup.find_all("a", href=True):
            label = " ".join(a.get_text(" ", strip=True).lower().split())
            if "securities" in label and "equity" in label and a["href"].lower().endswith(".csv"):
                candidates.append(requests.compat.urljoin(NSE_SECURITIES_PAGE, a["href"]))
    if not candidates:
        raise RuntimeError("Could not locate NSE's 'Securities available for Equity segment' CSV link.")
    raw = s.get(candidates[0], timeout=60)
    raw.raise_for_status()
    # NSE CSV may be UTF-8 with BOM; normalize column labels.
    data = pd.read_csv(io.BytesIO(raw.content), encoding_errors="replace")
    data.columns = [re.sub(r"\s+", " ", str(c).replace("\ufeff", "")).strip().upper() for c in data.columns]
    symbol_col = next((c for c in data.columns if c in {"SYMBOL", "SYMBOL "} or "SYMBOL" == c.strip()), None)
    series_col = next((c for c in data.columns if "SERIES" in c), None)
    if not symbol_col or not series_col:
        raise RuntimeError(f"Unexpected NSE equity-list columns: {data.columns.tolist()}")
    data[symbol_col] = data[symbol_col].astype(str).str.strip().str.upper()
    data[series_col] = data[series_col].astype(str).str.strip().str.upper()
    data = data[data[series_col].isin(MAINBOARD_SERIES)]
    data = data[data[symbol_col].str.match(r"^[A-Z0-9&._-]+$", na=False)]
    data = data.drop_duplicates(subset=[symbol_col]).sort_values(symbol_col)
    result = data[[symbol_col, series_col]].rename(columns={symbol_col: "Symbol", series_col: "Series"})
    if len(result) < 100:
        raise RuntimeError(f"NSE universe unexpectedly small ({len(result)} rows); refusing to publish incomplete file.")
    LOG.info("Loaded %d mainboard symbols from NSE", len(result))
    return result.reset_index(drop=True)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((loss == 0) & (gain > 0), 100)
    out = out.mask((gain == 0) & (loss > 0), 0)
    return out

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    return pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)

def adx_dmi(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14):
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr = true_range(h, l, c)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean(), plus, minus

def psar(h: pd.Series, l: pd.Series, step: float = 0.02, max_step: float = 0.20) -> pd.Series:
    """Classic Wilder parabolic SAR."""
    high, low = h.to_numpy(float), l.to_numpy(float)
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=h.index)
    bull = (high[1] + low[1]) >= (high[0] + low[0])
    sar = low[0] if bull else high[0]
    ep = high[0] if bull else low[0]
    af = step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i-1], low[i-2] if i > 1 else low[i-1])
            if low[i] < sar:
                bull, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            sar = max(sar, high[i-1], high[i-2] if i > 1 else high[i-1])
            if high[i] > sar:
                bull, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        out[i] = sar
    out[0] = low[0] if bull else high[0]
    return pd.Series(out, index=h.index)

def candle_label(o: float, h: float, l: float, c: float) -> str:
    body = abs(c-o)
    rng = max(h-l, 1e-12)
    upper = h-max(o,c)
    lower = min(o,c)-l
    if body/rng < 0.08:
        return "Doji"
    if lower > body*2 and upper < body:
        return "Hammer" if c >= o else "Hanging Man"
    if upper > body*2 and lower < body:
        return "Inverted Hammer" if c >= o else "Shooting Star"
    if c > o:
        return "Bullish"
    if c < o:
        return "Bearish"
    return "Doji"

def calculate_row(symbol: str, hist: pd.DataFrame) -> Optional[dict]:
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    hist = hist[~hist.index.duplicated(keep="last")].sort_index()
    if len(hist) < 200:
        LOG.warning("%s skipped: only %d usable sessions (need >=200)", symbol, len(hist))
        return None
    o, h, l, c, v = (hist[x].astype(float) for x in ["Open", "High", "Low", "Close", "Volume"])
    row = {"Symbol": symbol}
    row["Last Close"] = c.iloc[-1]
    row["Last Volume"] = v.iloc[-1]
    row["Avg Vol"] = v.tail(20).mean()
    for label, lag in [("% Chg 1 Day",1), ("% Chg 1 Week",5), ("% Chg 1 Month",21),
                       ("% Chg 3 Months",63), ("% Chg 1 Year",252)]:
        row[label] = (c.iloc[-1] / c.iloc[-1-lag] - 1) * 100 if len(c) > lag and c.iloc[-1-lag] else np.nan
    for p in [5,13,20,34,50,89,200]:
        row[f"SMA {p}"] = c.rolling(p, min_periods=p).mean().iloc[-1]
        row[f"EMA {p}"] = ema(c, p).iloc[-1]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    row["BBand Upper"], row["BBand Lower"], row["BBand Mid"] = (mid+2*sd).iloc[-1], (mid-2*sd).iloc[-1], mid.iloc[-1]
    macd = ema(c,12)-ema(c,26)
    signal = ema(macd,9)
    row["MACD"], row["MACD Signal"], row["MACD Hist"] = macd.iloc[-1], signal.iloc[-1], (macd-signal).iloc[-1]
    adx, dip, dim = adx_dmi(h,l,c)
    row["ADX"], row["DMI+"], row["DMI-"] = adx.iloc[-1], dip.iloc[-1], dim.iloc[-1]
    r = rsi(c)
    row["RSI"] = r.iloc[-1]
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = 100*(c-ll)/(hh-ll).replace(0,np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    row["Stoch %K"], row["Stoch %D"] = stoch_k.iloc[-1], stoch_d.iloc[-1]
    tp = (h+l+c)/3
    tp_sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x-np.mean(x))), raw=True)
    row["CCI"] = ((tp-tp_sma)/(0.015*mad.replace(0,np.nan))).iloc[-1]
    row["PSAR"] = psar(h,l).iloc[-1]
    tr = true_range(h,l,c)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    row["True Range"], row["ATR"] = tr.iloc[-1], atr.iloc[-1]
    row["Will %R"] = (-100*(hh-c)/(hh-ll).replace(0,np.nan)).iloc[-1]
    # TRIX: 15-period triple EMA percentage change.
    t3 = ema(ema(ema(c,15),15),15)
    row["TRIX"] = (t3.pct_change()*100).iloc[-1]
    rmin, rmax = r.rolling(14).min(), r.rolling(14).max()
    srsi = 100*(r-rmin)/(rmax-rmin).replace(0,np.nan)
    srsi_k, srsi_d = srsi.rolling(3).mean(), srsi.rolling(3).mean().rolling(3).mean()
    row["StochRSI %K"], row["StochRSI %D"] = srsi_k.iloc[-1], srsi_d.iloc[-1]
    row["Momentum"] = c.diff(10).iloc[-1]
    row["Candle"] = candle_label(o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1])
    row["Date"] = pd.Timestamp(hist.index[-1]).strftime("%Y-%m-%d")
    return row

def extract_ticker_frame(download: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if download is None or download.empty:
        return None
    if isinstance(download.columns, pd.MultiIndex):
        # yfinance normally returns (Price, Ticker) with group_by="column".
        if ticker in download.columns.get_level_values(-1):
            frame = download.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in download.columns.get_level_values(0):
            frame = download.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            return None
    else:
        frame = download.copy()
    required = {"Open","High","Low","Close","Volume"}
    if not required.issubset(set(frame.columns)):
        return None
    return frame

def download_and_build(universe: pd.DataFrame) -> pd.DataFrame:
    symbols = universe["Symbol"].tolist()
    tickers = [f"{s}.NS" for s in symbols]
    rows = []
    # Batch requests to avoid URL limits; 2y history provides 252-session returns + warmup.
    for start in range(0, len(tickers), 40):
        batch = tickers[start:start+40]
        LOG.info("Downloading price history %d-%d of %d", start+1, min(start+len(batch),len(tickers)),len(tickers))
        try:
            data = yf.download(batch, period="2y", interval="1d", auto_adjust=True,
                               group_by="column", threads=True, progress=False,
                               prepost=False, timeout=30)
        except Exception as exc:
            LOG.exception("Batch download failed: %s", exc)
            continue
        for ticker in batch:
            symbol = ticker[:-3]
            frame = extract_ticker_frame(data, ticker)
            if frame is None or frame.empty:
                LOG.warning("%s unavailable from price provider", symbol)
                continue
            try:
                row = calculate_row(symbol, frame)
                if row:
                    rows.append(row)
            except Exception:
                LOG.exception("Indicator calculation failed for %s", symbol)
        time.sleep(1)
    result = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in result.columns:
            result[col] = np.nan
    result = result[COLUMNS].sort_values("Symbol").reset_index(drop=True)
    # Preserve blank cells for unavailable indicators; round numeric columns consistently.
    for col in result.columns:
        if col not in {"Symbol","Candle","Date"}:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(2)
    return result

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = fetch_universe()
    result = download_and_build(universe)
    if result.empty:
        raise RuntimeError("No stock data was produced; output not written.")
    # Fail closed if the provider returns a suspiciously small subset.
    coverage = len(result) / len(universe)
    LOG.info("Coverage: %d/%d (%.1f%%)", len(result), len(universe), coverage*100)
    if coverage < 0.70:
        raise RuntimeError(f"Only {coverage:.1%} of NSE universe downloaded; refusing to replace output.")
    # Name the archive using the actual EOD session represented in the data,
    # not the calendar date on which the morning workflow happens to run.
    session_dates = pd.to_datetime(result["Date"], errors="coerce").dropna()
    if session_dates.empty:
        raise RuntimeError("No valid trading-session dates in output.")
    eod_date = session_dates.max().strftime("%Y-%m-%d")
    # Require a single common EOD date to avoid silently mixing sessions.
    date_counts = session_dates.dt.strftime("%Y-%m-%d").value_counts()
    if len(date_counts) > 1:
        LOG.warning("Output contains multiple last-session dates: %s", date_counts.head(10).to_dict())
    latest = OUTPUT_DIR / "nse_mainboard_eod.csv"
    dated = OUTPUT_DIR / f"nse_mainboard_eod_{eod_date}.csv"
    result.to_csv(latest, index=False, na_rep="")
    result.to_csv(dated, index=False, na_rep="")
    LOG.info("Wrote %s rows x %s columns to %s", len(result), len(COLUMNS), latest)
    print(f"OUTPUT={latest};ROWS={len(result)};COLUMNS={len(COLUMNS)};EOD_DATE={eod_date}")

if __name__ == "__main__":
    main()
