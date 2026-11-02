from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Params:
    vol_ratio_threshold: float = 1.45
    drawdown_flag_threshold: float = -0.12
    extension_threshold: float = 2.25
    compression_rank_threshold: float = 0.30
    range_rank_threshold: float = 0.55
    entry_toxic_max: int = 1
    exit_toxic_min: int = 4
    hard_crash_drawdown_threshold: float = -0.15
    hard_crash_ret3_threshold: float = -0.10
    min_hold_days: int = 3
    constructive_score_min: int = 3
    min_state_count: int = 15

def load_prices(source) -> pd.DataFrame:
    raw = pd.read_csv(source)
    date = next((c for c in ['Date','date'] if c in raw), None)
    close = next((c for c in ['TQQQ Close','Close Price','close','Close'] if c in raw), None)
    if not date or not close:
        raise ValueError('CSV requires Date and TQQQ Close or Close Price')
    mapping = {date:'date',close:'close', 'QQQ Close':'qqq_close','SOXX Close':'soxx_close','SPY Close':'spy_close'}
    df = raw.rename(columns={k:v for k,v in mapping.items() if k in raw})
    cols = ['date','close']+[x for x in ['qqq_close','soxx_close','spy_close'] if x in df]
    df = df[cols].copy()
    df['date'] = pd.to_datetime(df.date, errors='coerce')
    for c in cols[1:]: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['date','close']).sort_values('date').reset_index(drop=True)
    if df.date.duplicated().any(): raise ValueError('Duplicate dates found')
    if (df.close <= 0).any(): raise ValueError('Non-positive prices found')
    if len(df)<100: raise ValueError('At least 100 daily observations required')
    return df

def rank_last(a):
    return float(pd.Series(a).rank(pct=True).iloc[-1])

def features(df:pd.DataFrame) -> pd.DataFrame:
    x=df.copy(); p=x.close
    x['simple_ret']=p.pct_change().fillna(0.0)
    for n in (3,5,10): x[f'ret_{n}d']=np.log(p/p.shift(n))
    for n in (8,21,63): x[f'ema_{n}']=p.ewm(span=n,adjust=False,min_periods=n).mean()
    x['ema21_slope']=x.ema_21.pct_change()
    logret=np.log(p/p.shift(1))
    v5=logret.rolling(5).std(ddof=0);v20=logret.rolling(20).std(ddof=0)
    x['vol_ratio_5_20']=v5/v20.replace(0,np.nan)
    x['drawdown_20']=p/p.rolling(20).max()-1
    dist=p/x.ema_21-1
    def z(s):
        mu=s.rolling(126,min_periods=40).mean(); sd=s.rolling(126,min_periods=40).std(ddof=0)
        return (s-mu)/sd.replace(0,np.nan)
    x['extension_score']=z(dist)+.75*z(x.ret_5d)+.5*z(x.ret_10d)
    x['compression_rank_126']=x.vol_ratio_5_20.rolling(126,min_periods=40).apply(rank_last,raw=True)
    x['range_rank_126']=x.simple_ret.abs().rolling(126,min_periods=40).apply(rank_last,raw=True)
    if {'soxx_close','qqq_close'}.issubset(x.columns):
        x['soxx_vs_qqq_3d']=np.log(x.soxx_close/x.soxx_close.shift(3))-np.log(x.qqq_close/x.qqq_close.shift(3))
        x['leadership_available']=x.soxx_vs_qqq_3d.notna()
    else:
        x['soxx_vs_qqq_3d']=np.nan;x['leadership_available']=False
    x['trend_bucket']=np.select([dist>.0075,dist<-.015],['UP','DOWN'],default='MID')
    x['shock_bucket']=np.select([x.ret_3d<-.08,x.ret_3d>.05],['SHOCK_DN','SHOCK_UP'],default='NEUTRAL')
    # Vol bucket depends on candidate threshold, calculated in signals.
    return x

def state_history(x:pd.DataFrame, min_count:int) -> pd.DataFrame:
    # At close t, a five-day outcome for historical observation i is known only when i+5 <= t.
    n=len(x); hist_count=np.zeros(n,dtype=int); avg3=np.full(n,np.nan);avg5=np.full(n,np.nan);hit3=np.full(n,np.nan)
    stats={}
    for t in range(n):
        i=t-5
        if i>=0 and np.isfinite(x.iloc[i]['fwd_5d']) and np.isfinite(x.iloc[i]['fwd_3d']):
            r=x.iloc[i];k=r.state_key
            s=stats.setdefault(k,[0,0.,0.,0])
            s[0]+=1;s[1]+=float(r.fwd_3d);s[2]+=float(r.fwd_5d);s[3]+=float(r.fwd_3d>0)
        s=stats.get(x.iloc[t].state_key)
        if s:
            hist_count[t]=s[0];avg3[t]=s[1]/s[0];avg5[t]=s[2]/s[0];hit3[t]=s[3]/s[0]
    invest=(hist_count>=min_count)&(((avg3>.02)&(avg5>.03)&(hit3>.58))|((avg3>.005)&(hit3>=.5)))
    return pd.DataFrame({'hist_count':hist_count,'hist_avg_fwd_3d':avg3,'hist_avg_fwd_5d':avg5,'hist_hit_3d':hit3,'hist_invest_flag':invest.astype(int)},index=x.index)

def signals(base:pd.DataFrame, p:Params) -> pd.DataFrame:
    x=base.copy()
    x['vol_bucket']=np.select([x.vol_ratio_5_20>p.vol_ratio_threshold,x.vol_ratio_5_20<.9],['VOL_EXPAND','VOL_COMP'],default='VOL_MID')
    leader=np.select([x.soxx_vs_qqq_3d>-.01,x.soxx_vs_qqq_3d<-.02],['SEMI_OK','SEMI_LAG'],default='SEMI_MID')
    x['leadership_bucket']=np.where(x.leadership_available,leader,'SEMI_UNKNOWN')
    x['state_key']=x.trend_bucket+'|'+x.shock_bucket+'|'+x.vol_bucket+'|'+x.leadership_bucket
    x['fwd_3d']=np.log(x.close.shift(-3)/x.close)
    x['fwd_5d']=np.log(x.close.shift(-5)/x.close)
    hist=state_history(x,p.min_state_count)
    x=pd.concat([x,hist],axis=1)
    leader_bad=(x.soxx_vs_qqq_3d<-.025).fillna(False)
    x['toxic_flags']=((x.close<x.ema_63).astype(int)+(x.ema21_slope<0).astype(int)+(x.vol_ratio_5_20>p.vol_ratio_threshold).astype(int)+leader_bad.astype(int)+(x.drawdown_20<p.drawdown_flag_threshold).astype(int))
    x['constructive_score']=((x.close>x.ema_21).astype(int)+(x.ema21_slope>0).astype(int)+(x.ret_5d>0).astype(int)+((x.soxx_vs_qqq_3d>-.01)&x.leadership_available).astype(int))
    x['hard_crash_trigger']=(x.drawdown_20<p.hard_crash_drawdown_threshold)&(x.ret_3d<p.hard_crash_ret3_threshold)
    x['meltup_failure']=(x.extension_score>p.extension_threshold)&(x.compression_rank_126<p.compression_rank_threshold)&(x.range_rank_126<p.range_rank_threshold)&(x.close<x.ema_8)
    targets=[];held=0;days=0
    for r in x.itertuples():
        ready=np.isfinite(r.ema_63) and np.isfinite(r.ema_21)
        emergency=bool(r.hard_crash_trigger)
        must_exit=emergency or r.toxic_flags>=p.exit_toxic_min or bool(r.meltup_failure)
        enter=ready and not emergency and r.toxic_flags<=p.entry_toxic_max and r.constructive_score>=p.constructive_score_min and r.hist_invest_flag==1
        if not ready: held=0;days=0
        elif held:
            days+=1
            if emergency or (must_exit and days>=p.min_hold_days): held=0;days=0
        elif enter: held=1;days=1
        targets.append(held)
    x['target_position']=targets
    # Signals at t close can affect t+1 close-to-close returns only (idealized next-close execution).
    x['position']=x.target_position.shift(1).fillna(0).astype(float)
    return x

def apply_returns(x:pd.DataFrame, fee_bps:float=0, cash_rate:float=0)->pd.DataFrame:
    x=x.copy();r=x.simple_ret.fillna(0)
    turnover=x.position.diff().abs().fillna(x.position.abs())
    x['turnover']=turnover
    x['strategy_return']=x.position*r+(1-x.position)*(cash_rate/252)-turnover*(fee_bps/10000)
    x['buyhold_return']=r
    x['strategy_equity']=(1+x.strategy_return).cumprod()
    x['buyhold_equity']=(1+r).cumprod()
    return x

def metrics(ret:pd.Series)->dict:
    r=pd.Series(ret).fillna(0).astype(float)
    eq=(1+r).cumprod(); peak=np.maximum.accumulate(np.r_[1.,eq.to_numpy()]);dd=np.r_[1.,eq.to_numpy()]/peak-1
    n=len(r);sd=r.std(ddof=0)
    return {'cagr':float(eq.iloc[-1]**(252/n)-1),'max_drawdown':float(dd.min()),'sharpe':float(np.sqrt(252)*r.mean()/sd) if sd>0 else 0.,'volatility':float(sd*np.sqrt(252)),'total_return':float(eq.iloc[-1]-1)}

def evaluate(base,p,fee_bps=0,cash_rate=0):
    return apply_returns(signals(base,p),fee_bps,cash_rate)

def benchmarks(bt,fee_bps=0,cash_rate=0):
    r=bt.simple_ret.fillna(0);exposure=float(bt.position.mean())
    fixed=exposure*r+(1-exposure)*cash_rate/252
    ema_target=((bt.close>bt.ema_63)&bt.ema_63.notna()).astype(float)
    ema_position=ema_target.shift(1).fillna(0)
    ema=ema_position*r+(1-ema_position)*cash_rate/252-ema_position.diff().abs().fillna(0)*fee_bps/10000
    return {'Strategy':metrics(bt.strategy_return),'Buy and hold':metrics(r),'Exposure matched':metrics(fixed),'EMA63 timing':metrics(ema)}
