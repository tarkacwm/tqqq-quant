# TQQQ Research Foundation (release 1)

A GitHub/Streamlit Community Cloud-ready research app. The legacy pipeline is retained as `legacy_pipeline.py`; it is NOT used by the corrected engine.

## Deploy
1. Upload all repository files to GitHub, keeping `app.py`, `requirements.txt`, and `research/` at the root.
2. Open https://share.streamlit.io/ and create app from GitHub repository, branch `main`, entrypoint `app.py`.
3. Upload `data/sample_tqqq.csv` in the app to test.
4. Use the Baseline and Optimization tabs. Download trials CSV after optimization.

## Local run
`pip install -r requirements.txt && streamlit run app.py`

## Important limitations
- Sample CSV contains only TQQQ OHLCV; the original model relies on QQQ/SOXX leadership. With missing leadership data, this model deliberately does NOT invent neutral positive leadership; scores and state histories differ from a complete cross-asset dataset.
- Signal at close t is attributed to the close-to-close return t to t+1; this assumes exposure can be established at the signal close and is optimistic if the close is already known before the order. For executable research, model next-open entry and exit with OHLC data.
- Optuna optimizes on first 80% and scores one holdout once. This is a **foundation**, not a complete nested walk-forward system. Repeatedly selecting candidates by holdout performance contaminates the holdout.
- The optimizer's max drawdown constraint is a training constraint, not a guarantee. Optuna objective may select a penalized candidate if none pass.
- Transaction costs are simplified, with per-change bps. Cash interest defaults to zero. Source data must be split-adjusted and validated separately.
- No broker execution or live signal automation is included.

## Multi-ticker research (new)

The Streamlit sidebar supports **Online ticker download** (Yahoo Finance through
`yfinance`) for TQQQ, QQQ, SPY, SOXL, individual stocks, and supported crypto
symbols, plus **Upload price CSV**. Enter a ticker and date range, then load data.
Online downloads use auto-adjusted daily close prices; Yahoo data availability,
licensing and adjustment conventions may differ from YCharts. Internet access is
required for online downloads. The app caches the most recent downloaded symbol
in its session and clears prior optimization results when the selected dataset
changes.

CSV uploads support a single asset (`Date` plus `Close`, `Close Price`, `Adj Close`
or `TQQQ Close`) or long-form multiple assets (`Date`, `Ticker`, `Close`). The
selected ticker is filtered before features are calculated. For full leadership
features, include `QQQ Close` and `SOXX Close` columns in the selected asset's CSV.

**Important:** Strategy thresholds and shock/crash levels were developed for
leveraged TQQQ. Other tickers can be researched, but results are exploratory;
optimize and validate per instrument. The current benchmark is the *selected
asset*, not TQQQ. Online price data does not automatically include cross-asset
QQQ/SOXX inputs. Optimization results are not portable between assets.
