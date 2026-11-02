from __future__ import annotations
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
from .core import Params,evaluate,metrics

def run_search(base, trials=30, fee_bps=2., cash_rate=0., seed=42, min_dd=-.55):
    try: import optuna
    except ImportError as exc: raise RuntimeError('Install requirements.txt to use Optuna') from exc
    # Chronological holdout; optimize ONLY on training data. Holdout is not used in objective.
    cutoff=int(len(base)*.8)
    if cutoff<250 or len(base)-cutoff<50: raise ValueError('Need at least 300 rows')
    train=base.iloc[:cutoff].copy(); holdout=base.iloc[cutoff:].copy()
    def objective(trial):
        p=Params(vol_ratio_threshold=trial.suggest_float('vol_ratio_threshold',1.15,1.85),
          drawdown_flag_threshold=trial.suggest_float('drawdown_flag_threshold',-.19,-.08),
          entry_toxic_max=trial.suggest_int('entry_toxic_max',0,2),
          exit_toxic_min=trial.suggest_int('exit_toxic_min',2,5),
          hard_crash_drawdown_threshold=trial.suggest_float('hard_crash_drawdown_threshold',-.24,-.10),
          hard_crash_ret3_threshold=trial.suggest_float('hard_crash_ret3_threshold',-.15,-.06),
          min_hold_days=trial.suggest_int('min_hold_days',1,7),
          constructive_score_min=trial.suggest_int('constructive_score_min',2,4))
        bt=evaluate(train,p,fee_bps,cash_rate)
        m=metrics(bt.strategy_return)
        trial.set_user_attr('train_cagr',m['cagr']);trial.set_user_attr('train_drawdown',m['max_drawdown']);trial.set_user_attr('train_exposure',float(bt.position.mean()))
        if m['max_drawdown']<min_dd: return -10 + m['max_drawdown']
        # Mild penalty for trading, avoiding zero-exposure solutions.
        return m['cagr'] - .005*float(bt.turnover.sum())/(len(bt)/252)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective,n_trials=int(trials),show_progress_bar=False)
    records=[]
    for t in study.trials:
        records.append({'trial':t.number,'objective':t.value,**t.user_attrs,**t.params})
    results=pd.DataFrame(records).sort_values('objective',ascending=False)
    best=replace(Params(),**study.best_params)
    # Recompute on FULL history with frozen parameters; score only holdout segment.
    full=evaluate(base,best,fee_bps,cash_rate)
    test=full.iloc[cutoff:]
    test_metrics=metrics(test.strategy_return)
    return results,best,{'train_end':str(base.iloc[cutoff-1].date.date()),'holdout_start':str(base.iloc[cutoff].date.date()),'holdout_end':str(base.iloc[-1].date.date()),'holdout_metrics':test_metrics,'holdout_exposure':float(test.position.mean())},full
