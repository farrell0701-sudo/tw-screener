import pandas as pd, numpy as np, time
import v3c_opt as M
H='/home/claude/hdata'
lst=pd.read_csv(f'{H}/stock_list.csv',dtype={'code':str}); d={}
for c in lst.code:
    try: x=pd.read_csv(f'{H}/{c}.csv',parse_dates=['Date']).set_index('Date'); d[c]=x[x.index<'2026-10-16']
    except FileNotFoundError: pass
w=lambda k: pd.DataFrame({c:x[k] for c,x in d.items()}).sort_index()
raw,adj,hi,lo,vol,op=[w(k) for k in ('Close','AdjClose','High','Low','Volume','Open')]
hi=np.fmax(np.fmax(hi,op),raw); lo=np.fmin(np.fmin(lo,op),raw)
ix=pd.read_csv(f'{H}/TAIEX.csv',parse_dates=['Date']).set_index('Date').Close; ix=ix[ix.index<'2026-10-16']
M.DATA_START='2015-01-01'
P=M.build_panel(raw,adj,hi,lo,vol,ix,lst.set_index('code').industry.to_dict())
for dl in (0,1):
    t0=time.time(); E,T=M.simulate(P,dict(M.PRO,exec_delay=dl),'2016-01-01','2099-12-31')
    print('delay',dl,{k:round(float(v),3) for k,v in M.stats(E,T).items()},round(time.time()-t0,1),'s')
E,T=M.simulate(P,M.USER,'2021-01-01','2099-12-31'); print('USER 2021+',{k:round(float(v),3) for k,v in M.stats(E,T).items()})
import sys
if len(sys.argv)>1:
    M.TRAIN=('2016-06-01','2020-12-31'); M.TEST=('2021-01-01','2099-12-31'); M.N_STARTS=1
    M.run_all(P,n_random=2)
