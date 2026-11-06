# NSE Mainboard EOD Scanner

Creates one consolidated daily CSV for the NSE mainboard equity universe using the **exact 46-column schema** supplied in the original sample.

## What it does

1. Reads the official NSE "Securities available for Trading – Equity segment" list.
2. Keeps mainboard equity series (`EQ`, `BE`, `BZ`, `BT`, `ST`); SME is a separate NSE list and is excluded.
3. Downloads up to two years of daily OHLCV history per symbol from Yahoo Finance (`SYMBOL.NS`).
4. Calculates the technical indicators and writes:
   - `output/nse_mainboard_eod.csv` (latest file)
   - `output/nse_mainboard_eod_YYYY-MM-DD.csv` (dated using the actual latest session date in the data, not the run date)

The output contains exactly the following 46 columns, in the original order:

`Symbol, Last Close, Last Volume, Avg Vol, % Chg 1 Day, % Chg 1 Week, % Chg 1 Month, % Chg 3 Months, % Chg 1 Year, SMA 5, EMA 5, SMA 13, EMA 13, SMA 20, EMA 20, SMA 34, EMA 34, SMA 50, EMA 50, SMA 89, EMA 89, SMA 200, EMA 200, BBand Upper, BBand Lower, BBand Mid, MACD, MACD Signal, MACD Hist, ADX, DMI+, DMI-, RSI, Stoch %K, Stoch %D, CCI, PSAR, True Range, ATR, Will %R, TRIX, StochRSI %K, StochRSI %D, Momentum, Candle, Date`

## Set up daily generation on GitHub (free for public repositories)

1. Create a GitHub repository, preferably public if you want to use GitHub Actions free-tier minutes without consuming private-repository allowances.
2. Upload all files in this folder to the repository root. Keep `.github/workflows/daily-scan.yml` at exactly that path.
3. In the repository, open **Settings → Actions → General** and ensure workflows are permitted.
4. Open **Actions → NSE Mainboard EOD Scanner → Run workflow** for the first run.
5. After the first successful run, the workflow runs Monday–Friday at 17:15 India time (subject to GitHub Actions scheduling delays).
6. The generated files appear in the repository's `output/` folder. Download `nse_mainboard_eod.csv` whenever you need the latest scan.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python scanner.py
```

## Important data and methodology notes

- NSE's official equity-segment securities list defines the universe. The NSE page separates the SME list from the equity-segment list.
- Historical OHLCV is currently sourced from Yahoo Finance through `yfinance`, not directly from NSE. It is an unofficial convenience data source; coverage, symbol mapping, adjusted prices, and availability can vary. The script logs skipped symbols and stops if coverage falls below 70%.
- Prices use Yahoo Finance's auto-adjusted OHLC data. Thus historical indicator values and `Last Close` are adjusted values, which may differ from the unadjusted close in NSE's official bhavcopy.
- Indicator definitions are explicitly implemented in `scanner.py` (MACD 12/26/9, RSI 14, ADX/DMI 14, Bollinger 20 periods/2 standard deviations, etc.). Different charting platforms may use slightly different smoothing conventions.
- A stock with fewer than 200 available sessions is omitted because the sample requires 200-day moving averages and one-year change. To include newer listings with blanks in long-period fields, change the `len(hist) < 200` check in `calculate_row`.
- This is a technical-data scanner, not investment advice. Validate prices and indicators before trading.

## Troubleshooting

- If NSE changes its securities-list page or download link, `fetch_universe()` may need updating.
- If Yahoo Finance rate-limits or temporarily omits tickers, rerun the workflow. It refuses to overwrite the latest CSV when coverage is below 70%.
- GitHub scheduled workflows are not guaranteed to start at the exact minute. The schedule is a target time, not a real-time guarantee.
