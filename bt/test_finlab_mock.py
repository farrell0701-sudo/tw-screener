import pandas as pd, numpy as np, time
import finlab_v3c_pro as F
H='/home/claude/hdata'
lst=pd.read_csv(f'{H}/stock_list.csv',dtype={'code':str})
d={}
for c in lst.code:
    try: x=pd.read_csv(f'{H}/{c}.csv',parse_dates=['Date']).set_index('Date')
    except FileNotFoundError: continue
    d[c]=x[x.index<'2026-10-16']
def wide(k): return pd.DataFrame({c:x[k] for c,x in d.items()}).sort_index()
raw,adj,hi,lo,vol=wide('Close'),wide('AdjClose'),wide('High'),wide('Low'),wide('Volume')
ix=pd.read_csv(f'{H}/TAIEX.csv',parse_dates=['Date']).set_index('Date').Close; ix=ix[ix.index<'2026-10-16']
e60=ix.ewm(span=60,adjust=False).mean(); mkt=(e60/e60.shift(20)-1)>0
ind=lst.set_index('code').industry.to_dict()
t0=time.time(); sig,score,atrp,adv,rescue=F.build_signals(raw,adj,hi,lo,vol,mkt,ind); print('signals',round(time.time()-t0),'s')
for st in ['2016-01-01','2021-01-01']:
    p=dict(F.P,trade_start=st)
    E,T,W,N=F.simulate(sig,score,atrp,adv,rescue,raw,adj,p)
    m=F.metrics(E,T,ix); print(st,{k:(round(float(v),4) if not isinstance(v,dict) else v) for k,v in m.items()})
    print('  weights max sum',W.sum(axis=1).max().round(3),'avg hold',N.mean().round(2))
