
#!/usr/bin/env python3
"""NSE Mainboard Previous-Session EOD Scanner."""

from __future__ import annotations

import io
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
LOG = logging.getLogger("NSE_SCANNER")

IST = ZoneInfo("Asia/Kolkata")
OUTPUT_DIR = Path("output")

MIN_COVERAGE = 0.95
BATCH_SIZE = 40
HISTORY_PERIOD = "2y"

NSE_PAGE = (
    "https://www.nseindia.com/static/market-data/"
    "securities-available-for-trading"
)

NSE_HOME = "https://www.nseindia.com/"

MAINBOARD_SERIES = {"EQ", "BE", "BZ", "BT", "ST"}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/126 Safari/537.36"
)

COLUMNS = [
    "Symbol",
    "Last Close",
    "Last Volume",
    "Avg Vol",
    "% Chg 1 Day",
    "% Chg 1 Week",
    "% Chg 1 Month",
    "% Chg 3 Months",
    "% Chg 1 Year",
    "SMA 5",
    "EMA 5",
    "SMA 13",
    "EMA 13",
    "SMA 20",
    "EMA 20",
    "SMA 34",
    "EMA 34",
    "SMA 50",
    "EMA 50",
    "SMA 89",
    "EMA 89",
    "SMA 200",
    "EMA 200",
    "BBand Upper",
    "BBand Lower",
    "BBand Mid",
    "MACD",
    "MACD Signal",
    "MACD Hist",
    "ADX",
    "DMI+",
    "DMI-",
    "RSI",
    "Stoch %K",
    "Stoch %D",
    "CCI",
    "PSAR",
    "True Range",
    "ATR",
    "Will %R",
    "TRIX",
    "StochRSI %K",
    "StochRSI %D",
    "Momentum",
    "Candle",
    "Date"
]


# ---------------------------------------------------------
# NSE UNIVERSE
# ---------------------------------------------------------

def get_nse_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json,*/*",
        "Referer": NSE_HOME,
        "Accept-Language": "en-US,en;q=0.9"
    })

    session.get(NSE_HOME, timeout=30)

    return session


def fetch_universe():

    session = get_nse_session()

    response = session.get(NSE_PAGE, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for link in soup.find_all("a", href=True):

        href = link["href"]
        label = link.get_text(" ", strip=True).lower()

        if ".csv" in href.lower():

            if (
                "equity" in label
                or "equity" in href.lower()
                or "securities" in label
            ):
                links.append(
                    requests.compat.urljoin(NSE_PAGE, href)
                )

    if not links:
        raise RuntimeError(
            "Could not locate NSE equity securities CSV. "
            "Check whether NSE has changed its download page."
        )

    csv_response = session.get(links[0], timeout=60)
    csv_response.raise_for_status()

    data = pd.read_csv(
        io.BytesIO(csv_response.content),
        encoding_errors="replace"
    )

    data.columns = [
        str(c).replace("\ufeff", "").strip().upper()
        for c in data.columns
    ]

    symbol_col = next(
        (c for c in data.columns if c.strip() == "SYMBOL"),
        None
    )

    series_col = next(
        (c for c in data.columns if "SERIES" in c),
        None
    )

    if symbol_col is None or series_col is None:
        raise RuntimeError(
            f"Unexpected NSE CSV format: {data.columns.tolist()}"
        )

    data[symbol_col] = (
        data[symbol_col].astype(str).str.strip().str.upper()
    )

    data[series_col] = (
        data[series_col].astype(str).str.strip().str.upper()
    )

    data = data[
        data[series_col].isin(MAINBOARD_SERIES)
    ]

    data = data[
        data[symbol_col].str.match(
            r"^[A-Z0-9&._-]+$", na=False
        )
    ]

    data = data.drop_duplicates(
        subset=[symbol_col]
    )

    result = data[[symbol_col]].rename(
        columns={symbol_col: "Symbol"}
    )

    result = result.sort_values("Symbol").reset_index(drop=True)

    if len(result) < 100:
        raise RuntimeError(
            f"NSE universe unexpectedly small: {len(result)}"
        )

    LOG.info("NSE mainboard universe: %d stocks", len(result))

    return result


# ---------------------------------------------------------
# TARGET PREVIOUS TRADING SESSION
# ---------------------------------------------------------

def get_target_session():

    today = datetime.now(IST).date()

    LOG.info("Today's IST date: %s", today)

    benchmark = yf.download(
        "^NSEI",
        period="30d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if benchmark is None or benchmark.empty:
        raise RuntimeError(
            "Cannot determine previous NSE session: "
            "NIFTY 50 history unavailable."
        )

    dates = pd.to_datetime(benchmark.index).date

    eligible = [
        d for d in dates if d < today
    ]

    if not eligible:
        raise RuntimeError(
            "No previous trading session found."
        )

    target = max(eligible)

    LOG.info("Target EOD session: %s", target)

    return target


# ---------------------------------------------------------
# INDICATOR FUNCTIONS
# ---------------------------------------------------------

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def sma(series, period):

    return series.rolling(
        period,
        min_periods=period
    ).mean()


def rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    result = result.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )

    return result


def true_range(high, low, close):

    previous_close = close.shift(1)

    return pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)


def calculate_adx(high, low, close, period=14):

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=high.index
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=high.index
    )

    tr = true_range(high, low, close)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean() / atr
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean() / atr
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return adx, plus_di, minus_di


def calculate_psar(high, low, step=0.02, maximum=0.20):

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)

    n = len(h)

    output = np.full(n, np.nan)

    if n == 0:
        return pd.Series(output, index=high.index)

    if n == 1:
        output[0] = l[0]
        return pd.Series(output, index=high.index)

    bullish = (h[1] + l[1]) >= (h[0] + l[0])

    sar = l[0] if bullish else h[0]
    ep = h[0] if bullish else l[0]

    af = step

    output[0] = sar

    for i in range(1, n):

        sar = sar + af * (ep - sar)

        if bullish:

            sar = min(
                sar,
                l[i - 1],
                l[i - 2] if i > 1 else l[i - 1]
            )

            if l[i] < sar:

                bullish = False
                sar = ep
                ep = l[i]
                af = step

            elif h[i] > ep:

                ep = h[i]
                af = min(af + step, maximum)

        else:

            sar = max(
                sar,
                h[i - 1],
                h[i - 2] if i > 1 else h[i - 1]
            )

            if h[i] > sar:

                bullish = True
                sar = ep
                ep = h[i]
                af = step

            elif l[i] < ep:

                ep = l[i]
                af = min(af + step, maximum)

        output[i] = sar

    return pd.Series(output, index=high.index)


def candle_name(o, h, l, c):

    body = abs(c - o)
    total = max(h - l, 1e-12)

    upper = h - max(o, c)
    lower = min(o, c) - l

    if body / total < 0.08:
        return "Doji"

    if lower > body * 2 and upper < body:
        return "Hammer" if c >= o else "Hanging Man"

    if upper > body * 2 and lower < body:
        return (
            "Inverted Hammer"
            if c >= o else "Shooting Star"
        )

    if c > o:
        return "Bullish"

    if c < o:
        return "Bearish"

    return "Doji"


# ---------------------------------------------------------
# CALCULATE ALL 46 FIELDS
# ---------------------------------------------------------

def calculate_row(symbol, history, target_session):

    history = history.copy()

    history = history.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    history = history[
        ~history.index.duplicated(keep="last")
    ].sort_index()

    if history.empty:
        return None

    last_date = pd.Timestamp(history.index[-1]).date()

    if last_date != target_session:
        return None

    o = history["Open"].astype(float)
    h = history["High"].astype(float)
    l = history["Low"].astype(float)
    c = history["Close"].astype(float)
    v = history["Volume"].astype(float)

    row = {"Symbol": symbol}

    row["Last Close"] = c.iloc[-1]
    row["Last Volume"] = v.iloc[-1]
    row["Avg Vol"] = v.tail(20).mean()

    # Returns
    periods = [
        ("% Chg 1 Day", 1),
        ("% Chg 1 Week", 5),
        ("% Chg 1 Month", 21),
        ("% Chg 3 Months", 63),
        ("% Chg 1 Year", 252)
    ]

    for name, lag in periods:

        if len(c) > lag and c.iloc[-1 - lag] != 0:

            row[name] = (
                c.iloc[-1] / c.iloc[-1 - lag] - 1
            ) * 100

        else:
            row[name] = np.nan

    # Moving averages
    for period in [5, 13, 20, 34, 50, 89, 200]:

        row[f"SMA {period}"] = sma(c, period).iloc[-1]
        row[f"EMA {period}"] = ema(c, period).iloc[-1]

    # Bollinger bands
    middle = sma(c, 20)
    deviation = c.rolling(20).std(ddof=0)

    row["BBand Upper"] = (middle + 2 * deviation).iloc[-1]
    row["BBand Lower"] = (middle - 2 * deviation).iloc[-1]
    row["BBand Mid"] = middle.iloc[-1]

    # MACD
    macd = ema(c, 12) - ema(c, 26)
    signal = ema(macd, 9)

    row["MACD"] = macd.iloc[-1]
    row["MACD Signal"] = signal.iloc[-1]
    row["MACD Hist"] = (macd - signal).iloc[-1]

    # ADX / DMI
    adx, plus_di, minus_di = calculate_adx(h, l, c)

    row["ADX"] = adx.iloc[-1]
    row["DMI+"] = plus_di.iloc[-1]
    row["DMI-"] = minus_di.iloc[-1]

    # RSI
    r = rsi(c)

    row["RSI"] = r.iloc[-1]

    # Stochastic
    lowest = l.rolling(14).min()
    highest = h.rolling(14).max()

    stoch_k = (
        100 * (c - lowest)
        / (highest - lowest).replace(0, np.nan)
    )

    stoch_d = stoch_k.rolling(3).mean()

    row["Stoch %K"] = stoch_k.iloc[-1]
    row["Stoch %D"] = stoch_d.iloc[-1]

    # CCI
    typical = (h + l + c) / 3

    typical_mean = typical.rolling(20).mean()

    mean_deviation = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    cci = (
        (typical - typical_mean)
        / (0.015 * mean_deviation.replace(0, np.nan))
    )

    row["CCI"] = cci.iloc[-1]

    # PSAR
    row["PSAR"] = calculate_psar(h, l).iloc[-1]

    # True Range / ATR
    tr = true_range(h, l, c)

    atr = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    row["True Range"] = tr.iloc[-1]
    row["ATR"] = atr.iloc[-1]

    # Williams %R
    row["Will %R"] = (
        -100 * (highest - c)
        / (highest - lowest).replace(0, np.nan)
    ).iloc[-1]

    # TRIX
    triple_ema = ema(ema(ema(c, 15), 15), 15)

    row["TRIX"] = (
        triple_ema.pct_change() * 100
    ).iloc[-1]

    # Stochastic RSI
    rsi_min = r.rolling(14).min()
    rsi_max = r.rolling(14).max()

    stoch_rsi = (
        100 * (r - rsi_min)
        / (rsi_max - rsi_min).replace(0, np.nan)
    )

    stoch_rsi_k = stoch_rsi.rolling(3).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(3).mean()

    row["StochRSI %K"] = stoch_rsi_k.iloc[-1]
    row["StochRSI %D"] = stoch_rsi_d.iloc[-1]

    # Momentum
    row["Momentum"] = c.diff(10).iloc[-1]

    # Candle
    row["Candle"] = candle_name(
        o.iloc[-1],
        h.iloc[-1],
        l.iloc[-1],
        c.iloc[-1]
    )

    row["Date"] = target_session.isoformat()

    return row


# ---------------------------------------------------------
# YAHOO DATA DOWNLOAD
# ---------------------------------------------------------

def extract_ticker_frame(data, ticker):

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):

        if ticker in data.columns.get_level_values(-1):

            frame = data.xs(
                ticker,
                axis=1,
                level=-1,
                drop_level=True
            )

        elif ticker in data.columns.get_level_values(0):

            frame = data.xs(
                ticker,
                axis=1,
                level=0,
                drop_level=True
            )

        else:
            return None

    else:
        frame = data.copy()

    required = {
        "Open", "High", "Low", "Close", "Volume"
    }

    if not required.issubset(frame.columns):
        return None

    return frame


def download_and_build(universe, target_session):

    symbols = universe["Symbol"].tolist()

    tickers = [
        f"{symbol}.NS" for symbol in symbols
    ]

    rows = []

    for start in range(0, len(tickers), BATCH_SIZE):

        batch = tickers[start:start + BATCH_SIZE]

        LOG.info(
            "Downloading batch %d of %d",
            start // BATCH_SIZE + 1,
            (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        )

        try:

            data = yf.download(
                batch,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="column",
                threads=True,
                progress=False,
                prepost=False,
                timeout=40
            )

        except Exception:

            LOG.exception("Batch download failed")
            continue

        for ticker in batch:

            symbol = ticker[:-3]

            frame = extract_ticker_frame(data, ticker)

            if frame is None or frame.empty:

                LOG.warning("%s unavailable", symbol)
                continue

            # CRITICAL:
            # Exclude every candle after the target EOD session.
            dates = pd.to_datetime(frame.index).date

            frame = frame.loc[
                [d <= target_session for d in dates]
            ]

            if frame.empty:

                LOG.warning("%s has no historical data", symbol)
                continue

            # Require exact target date. No stale stock rows.
            if pd.Timestamp(frame.index[-1]).date() != target_session:

                LOG.warning(
                    "%s missing target session %s",
                    symbol,
                    target_session
                )
                continue

            try:

                row = calculate_row(
                    symbol,
                    frame,
                    target_session
                )

                if row is not None:
                    rows.append(row)

            except Exception:

                LOG.exception(
                    "Indicator calculation failed for %s",
                    symbol
                )

        time.sleep(1)

    result = pd.DataFrame(rows)

    for column in COLUMNS:

        if column not in result.columns:
            result[column] = np.nan

    result = result[COLUMNS]

    result = result.sort_values(
        "Symbol"
    ).reset_index(drop=True)

    # Round numerical fields.
    for column in COLUMNS:

        if column not in {"Symbol", "Candle", "Date"}:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            ).round(2)

    return result


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_output(result, universe, target_session):

    if result.empty:
        raise RuntimeError("No stock data generated.")

    if result.columns.tolist() != COLUMNS:
        raise RuntimeError("CSV column structure mismatch.")

    if result["Symbol"].duplicated().any():
        raise RuntimeError("Duplicate stock symbols found.")

    dates = result["Date"].dropna().unique()

    if len(dates) != 1 or dates[0] != target_session.isoformat():

        raise RuntimeError(
            f"Invalid EOD dates found: {dates.tolist()}"
        )

    if result["Last Close"].isna().any():

        raise RuntimeError("Missing closing prices found.")

    expected = set(universe["Symbol"])
    actual = set(result["Symbol"])

    coverage = len(actual & expected) / len(expected)

    LOG.info(
        "Universe: %d | Retrieved: %d | Coverage: %.2f%%",
        len(expected),
        len(actual),
        coverage * 100
    )

    missing = sorted(expected - actual)

    LOG.warning(
        "Stocks not retrieved: %d",
        len(missing)
    )

    if coverage < MIN_COVERAGE:

        raise RuntimeError(
            f"Coverage {coverage:.2%} is below "
            f"the required {MIN_COVERAGE:.0%}. "
            "CSV will not be published."
        )

    LOG.info("Validation successful.")

    return coverage


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG.info("Starting NSE Mainboard EOD scanner.")

    # 1. NSE universe
    universe = fetch_universe()

    # 2. Determine previous completed session
    target_session = get_target_session()

    # 3. Download and calculate
    result = download_and_build(
        universe,
        target_session
    )

    # 4. Validate
    coverage = validate_output(
        result,
        universe,
        target_session
    )

    # 5. Publish only after successful validation
    latest = OUTPUT_DIR / "nse_mainboard_eod.csv"

    dated = OUTPUT_DIR / (
        f"nse_mainboard_eod_{target_session.isoformat()}.csv"
    )

    temporary = OUTPUT_DIR / "nse_mainboard_eod.tmp.csv"

    result.to_csv(
        temporary,
        index=False,
        na_rep=""
    )

    # Atomic replacement of latest file
    temporary.replace(latest)

    result.to_csv(
        dated,
        index=False,
        na_rep=""
    )

    LOG.info("-----------------------------------")
    LOG.info("SCAN COMPLETED SUCCESSFULLY")
    LOG.info("EOD DATE: %s", target_session)
    LOG.info("STOCKS: %d", len(result))
    LOG.info("COLUMNS: %d", len(result.columns))
    LOG.info("COVERAGE: %.2f%%", coverage * 100)
    LOG.info("LATEST FILE: %s", latest)
    LOG.info("ARCHIVE FILE: %s", dated)
    LOG.info("-----------------------------------")


if __name__ == "__main__":
    main()
