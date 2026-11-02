import io
import pandas as pd
import pytest
from research.data_sources import normalize_ticker, csv_tickers, load_ticker_csv

def test_symbol_validation():
    assert normalize_ticker(' btc-usd ') == 'BTC-USD'
    with pytest.raises(ValueError): normalize_ticker('../bad')

def test_long_csv_selects_ticker():
    dates=pd.date_range('2024-01-01',periods=110,freq='B')
    rows=[]
    for ticker,scale in [('TQQQ',1),('SPY',2)]:
        rows += [{'Date':str(d.date()),'Ticker':ticker,'Close':scale*(100+i)} for i,d in enumerate(dates)]
    buf=io.BytesIO(pd.DataFrame(rows).to_csv(index=False).encode())
    assert csv_tickers(buf)==['SPY','TQQQ']
    frame=load_ticker_csv(buf,'SPY')
    assert len(frame)==110
    assert frame.close.iloc[0]==200
    assert frame.close.iloc[-1]==418
