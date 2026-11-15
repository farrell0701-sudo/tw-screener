import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
rows=[]
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    V=pd.read_pickle(f'/home/claude/old/L102_{w}.pkl'); a,b=V['_window']; X=pd.read_pickle(f'/home/claude/old/L100P_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]
    M=[mix([p,q],[.6,.4]) for p,q in zip(g(X['W60q20']),g(V['FIB4']))]
    rows.append(pd.concat(M,axis=1).mean(axis=1))
st=pd.concat(rows); st=st[~st.index.duplicated()]
pd.to_pickle(st,'/home/claude/old/M091_strat.pkl'); print('saved strat', len(st))
