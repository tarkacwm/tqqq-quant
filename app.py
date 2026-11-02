import io,json
from dataclasses import asdict
import pandas as pd
import streamlit as st
from research import Params,load_prices,features,evaluate,benchmarks,run_search
st.set_page_config(page_title='TQQQ Research Foundation',layout='wide')
st.title('TQQQ | Research Foundation')
st.caption('Binary invested/cash research • close-of-day signals • next-session return attribution')
source=st.file_uploader('Upload YCharts CSV (TQQQ Close or Close Price)',type=['csv'])
if source is None:
    st.info('Upload a CSV to begin. The repository includes data/sample_tqqq.csv for testing.')
    st.stop()
try:
    raw=load_prices(source);base=features(raw)
except Exception as e:
    st.error(str(e));st.stop()
st.success(f'{len(raw):,} rows • {raw.date.iloc[0].date()} to {raw.date.iloc[-1].date()}')
if not {'qqq_close','soxx_close'}.issubset(raw.columns):
    st.warning('QQQ/SOXX prices are missing. Leadership is UNKNOWN and the constructive score loses one possible point. Results are not directly comparable to a full cross-asset dataset.')
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
    st.metric('Latest target (next session)', 'INVESTED' if bt.target_position.iloc[-1] else 'CASH')
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
        st.download_button('Download trials CSV',st.session_state['results'].to_csv(index=False),'optuna_trials.csv','text/csv')
        st.subheader('Best training configuration')
        st.json(st.session_state['best'])
        st.subheader('Untouched holdout report')
        st.json(st.session_state['holdout'])
        st.warning('Holdout is reported for evaluation, not for repeated model selection. For robust claims, add nested walk-forward validation.')
with t3:
    st.markdown('''**Corrected:** State forward returns enter history only five sessions after observation. Emergency crash exits bypass minimum hold. The daily position is lagged one bar. Returns are compounded from simple returns rather than log-return approximations.\n\n**Execution convention:** End-of-day signal at date *t* controls the return from close *t* to close *t+1*. This is an idealized next-session exposure assumption, **not** a verified executable close-to-close fill. A production execution simulator should model next-open fills and overnight gaps using OHLC data.\n\n**Still to validate:** Data adjustments and split handling; total-return convention; cross-asset alignment; transaction-cost realism; survivorship; data quality; and walk-forward optimization. The provided TQQQ-only CSV does not include QQQ/SOXX leadership.''')
