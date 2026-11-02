import numpy as np
import pandas as pd
from research.core import Params,features,signals,apply_returns,metrics,state_history

def fixture():
    d=pd.bdate_range('2020-01-01',periods=200)
    return features(pd.DataFrame({'date':d,'close':100*np.exp(np.arange(200)*.003)}))

def test_lag():
    x=signals(fixture(),Params())
    assert x.position.iloc[1:].reset_index(drop=True).equals(x.target_position.iloc[:-1].astype(float).reset_index(drop=True))

def test_history_matures_after_five_days():
    x=fixture();x['state_key']='S';x['fwd_3d']=np.log(x.close.shift(-3)/x.close);x['fwd_5d']=np.log(x.close.shift(-5)/x.close)
    h=state_history(x,1)
    assert h.hist_count.iloc[:5].eq(0).all()
    assert h.hist_count.iloc[5]==1
    assert h.hist_count.iloc[6]==2

def test_simple_compounding():
    x=pd.DataFrame({'simple_ret':[0,.1,-.1],'position':[0,1,1]})
    bt=apply_returns(x)
    assert abs(bt.strategy_equity.iloc[-1]-.99)<1e-10

def test_emergency_bypasses_hold():
    x=fixture();x['state_key']='S';x['fwd_3d']=.03;x['fwd_5d']=.04
    # Test state machine by constructing enough history, and forcing emergency at a held bar.
    p=Params(min_state_count=1,min_hold_days=7,constructive_score_min=2)
    x.loc[100:,'ret_5d']=.02
    x.loc[100:,'ema21_slope']=.01
    x.loc[100:,'ema_21']=x.close*.98
    x.loc[100:,'ema_63']=x.close*.95
    x.loc[100:,'vol_ratio_5_20']=1.
    x.loc[100:,'drawdown_20']=0.
    x.loc[150,'drawdown_20']=-.25;x.loc[150,'ret_3d']=-.2
    z=signals(x,p)
    assert z.target_position.iloc[149]==1
    assert z.target_position.iloc[150]==0
