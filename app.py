import io,json
from dataclasses import asdict
import pandas as pd
import streamlit as st
from research import Params,load_prices,features,evaluate,benchmarks,run_search
from datetime import date, timedelta
from research import normalize_ticker, csv_tickers, load_ticker_csv, download_prices
st.set_page_config(page_title='Multi-Ticker Research Foundation',layout='wide')
st.title('Quant Research | Multi-Ticker Foundation')
st.caption('Research any supported ticker • binary long/cash model • historical data validation')
with st.sidebar:
    st.header('Research universe')
    mode=st.radio('Data source',['Online ticker download','Upload price CSV'],help='Online downloads use Yahoo Finance via yfinance. CSV upload works if online data is unavailable.')
    if mode=='Online ticker download':
        symbol_input=st.text_input('Ticker symbol','TQQQ',help='Examples: TQQQ, QQQ, SPY, SOXL, AAPL, BTC-USD')
        start_date=st.date_input('Start date',date(2015,1,1),max_value=date.today())
        end_date=st.date_input('End date (inclusive)',date.today(),max_value=date.today())
        refresh=st.button('Load / refresh ticker data',type='primary')
    else:
        source=st.file_uploader('Upload daily price CSV',type=['csv'])
        available=[]
        if source is not None:
            try: available=csv_tickers(source)
            except Exception as exc: st.error(f'Could not inspect CSV: {exc}')
        if available:
            symbol_input=st.selectbox('Ticker in CSV',available)
        else:
            symbol_input=st.text_input('Ticker label for this CSV','TQQQ')
        refresh=False
try:
    symbol=normalize_ticker(symbol_input)
    if mode=='Online ticker download':
        if start_date>=end_date: raise ValueError('Start date must precede end date.')
        key=(symbol,str(start_date),str(end_date))
        if refresh or st.session_state.get('price_key')!=key:
            with st.spinner(f'Downloading adjusted daily history for {symbol}...'):
                st.session_state['price_data']=download_prices(symbol,str(start_date),str(end_date+timedelta(days=1)))
                st.session_state['price_key']=key
                st.session_state.pop('results',None)
        raw=st.session_state['price_data']
    else:
        if source is None:
            st.info('Upload a CSV or select Online ticker download to begin.')
            st.stop()
        raw=load_ticker_csv(source,symbol if available else None)
        upload_key=(source.name,source.size,symbol)
        if st.session_state.get('price_key')!=upload_key:
            st.session_state.pop('results',None)
            st.session_state['price_key']=upload_key
    base=features(raw)
except Exception as exc:
    st.error(f'Data load failed: {exc}')
    st.stop()
st.success(f'{symbol} • {len(raw):,} rows • {raw.date.iloc[0].date()} to {raw.date.iloc[-1].date()}')
st.warning('The current thresholds were designed around TQQQ. Applying them to other assets is exploratory and requires asset-specific calibration. Cross-asset leadership is unavailable unless your CSV includes QQQ Close and SOXX Close.')
if not {'qqq_close','soxx_close'}.issubset(raw.columns):
    st.info('QQQ/SOXX leadership is UNKNOWN for this dataset. The constructive score loses one possible point; do not compare directly with runs using full cross-asset inputs.')
with st.sidebar:
    st.header('Research assumptions')
    fee=st.number_input('Cost per position change (bps)',0.,50.,2.,.5)
    cash=st.number_input('Annual cash yield (decimal)',0.,.20,0.,.005,format='%.3f')
    trials=st.slider('Optuna trials',5,200,25,5)
    dd=st.slider('Train max drawdown floor',-.90,-.20,-.55,.05)
    st.caption('A cost is charged on changes in the position held for daily returns.')
t1,t2,t3=st.tabs(['Baseline & benchmarks','Optimization Lab','Audit notes'])
with t1:
    bt=evaluate(base,Params(),fee,cash)
    comparison=pd.DataFrame(benchmarks(bt,fee,cash)).T
    st.subheader('Baseline and benchmarks')
    st.dataframe(comparison.style.format('{:.2%}',subset=['cagr','max_drawdown','volatility','total_return']).format('{:.2f}',subset=['sharpe']),use_container_width=True)
    st.line_chart(bt.set_index('date')[['strategy_equity','buyhold_equity']])
    st.metric(f'{symbol} latest target (next session)', 'INVESTED' if bt.target_position.iloc[-1] else 'CASH')
    st.caption(f"Latest state: {bt.state_key.iloc[-1]} • historical state count: {int(bt.hist_count.iloc[-1])} • target is not the position attributed to today's return")
with t2:
    st.write('Optimization uses the first 80% of dates only. The final 20% is an untouched chronological holdout during trial selection.')
    if st.button('Run Optuna research',type='primary'):
        with st.spinner('Optimizing and evaluating frozen candidate...'):
            try: results,best,holdout,full=run_search(base,trials,fee,cash,42,dd)
            except Exception as e: st.error(str(e));st.stop()
        st.session_state['results']=results
        st.session_state['best']=asdict(best)
        st.session_state['holdout']=holdout
        st.session_state['optimized']=full
    if 'results' in st.session_state:
        st.subheader('Optimization results')
        st.dataframe(st.session_state['results'],use_container_width=True)
        st.download_button('Download trials CSV',st.session_state['results'].to_csv(index=False),f'{symbol}_optuna_trials.csv','text/csv')
        st.subheader('Best training configuration')
        st.json(st.session_state['best'])
        st.subheader('Untouched holdout report')
        st.json(st.session_state['holdout'])
        st.warning('Holdout is reported for evaluation, not for repeated model selection. For robust claims, add nested walk-forward validation.')
with t3:
    st.markdown('''**Corrected:** State forward returns enter history only five sessions after observation. Emergency crash exits bypass minimum hold. The daily position is lagged one bar. Returns are compounded from simple returns rather than log-return approximations.\n\n**Execution convention:** End-of-day signal at date *t* controls the return from close *t* to close *t+1*. This is an idealized next-session exposure assumption, **not** a verified executable close-to-close fill. A production execution simulator should model next-open fills and overnight gaps using OHLC data.\n\n**Still to validate:** Data adjustments and split handling; total-return convention; cross-asset alignment; transaction-cost realism; survivorship; data quality; and walk-forward optimization. The provided TQQQ-only CSV does not include QQQ/SOXX leadership.''')
