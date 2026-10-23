import pandas as pd, numpy as np, time
import ema12_opt as M
H='/home/claude/hdata'
lst=pd.read_csv(f'{H}/stock_list.csv',dtype={'code':str}); d={}
for c in lst.code:
    try: x=pd.read_csv(f'{H}/{c}.csv',parse_dates=['Date']).set_index('Date'); d[c]=x[x.index<'2026-10-16']
    except FileNotFoundError: pass
w=lambda k: pd.DataFrame({c:x[k] for c,x in d.items()}).sort_index()
raw_c,adj,hi,lo,vol,op=[w(k) for k in ('Close','AdjClose','High','Low','Volume','Open')]
hi=np.fmax(np.fmax(hi,op),raw_c); lo=np.fmin(np.fmin(lo,op),raw_c)
f=adj/raw_c
ix=pd.read_csv(f'{H}/TAIEX.csv',parse_dates=['Date']).set_index('Date').Close; ix=ix[ix.index<'2026-10-16']
P=M.build_panel(adj,hi*f,lo*f,raw_c*vol,raw_c,hi,lo,vol,ix,lst.set_index('code').industry.to_dict())
M.TRAIN=('2016-01-01','2020-12-31'); M.TEST=('2021-01-01','2099-12-31')
t0=time.time(); E,T=M.simulate(P,M.BASE_ORIG,'2021-01-01','2099-12-31'); print('一次模擬',round(time.time()-t0,1),'秒')
print('原始規則 2021起',{k:round(float(v),3) for k,v in M.stats(E,T).items()})
E,T=M.simulate(P,M.BASE_ORIG,'2016-01-01','2020-12-31'); print('原始規則 2016-2020',{k:round(float(v),3) for k,v in M.stats(E,T).items()})
import sys
if len(sys.argv)>1: M.N_STARTS=3; M.run_all(P,grid=False,n_random=5)
