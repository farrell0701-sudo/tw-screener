import pandas as pd, numpy as np, time
import finlab_v3c_pro as F, finlab_compare as C
H='/home/claude/hdata'
lst=pd.read_csv(f'{H}/stock_list.csv',dtype={'code':str}); d={}
for c in lst.code:
    try: x=pd.read_csv(f'{H}/{c}.csv',parse_dates=['Date']).set_index('Date')
    except FileNotFoundError: continue
    d[c]=x[x.index<'2026-10-16']
def wide(k): return pd.DataFrame({c:x[k] for c,x in d.items()}).sort_index()
raw,adj,hi,lo,vol,op=[wide(k) for k in ('Close','AdjClose','High','Low','Volume','Open')]
hi=np.fmax(np.fmax(hi,op),raw); lo=np.fmin(np.fmin(lo,op),raw)
ix=pd.read_csv(f'{H}/TAIEX.csv',parse_dates=['Date']).set_index('Date').Close; ix=ix[ix.index<'2026-10-16']
e60=ix.ewm(span=60,adjust=False).mean(); mkt=(e60/e60.shift(20)-1)>0
ind=lst.set_index('code').industry.to_dict()
td=pd.read_csv('/home/claude/farrell0701-sudo/tw-screener/data/tdcc/20261008.csv',dtype={'code':str})
cap=(td[td.level==17].set_index('code').shares*10/1e8).to_dict()
t0=time.time(); b=C.build_300(raw,adj,hi,lo,vol,mkt,ind,cap); print('build',round(time.time()-t0))
t0=time.time(); E,T,W,N=F.simulate(*b,raw,adj,dict(C.P300,trade_start='2016-01-01')); print('sim',round(time.time()-t0),'s')
m=F.metrics(E,T); print('300萬版',round(m['年化'],4),round(m['最大回撤'],4),m['交易數'])
E,T,W,N=F.simulate(*b,raw,adj,dict(C.P300,trade_start='2016-01-01',rank='random',seed=1)); print('random',round(F.metrics(E,T)['年化'],4))
