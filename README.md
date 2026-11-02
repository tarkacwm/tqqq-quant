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
