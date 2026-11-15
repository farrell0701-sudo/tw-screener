import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    V=pd.read_pickle(f'/home/claude/old/L102_{w}.pkl'); a,b=V['_window']; X=pd.read_pickle(f'/home/claude/old/L100P_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]; x=g(X['W60q20'])
    S[w]={}
    for k in ['FIB','FIB_S','FIB_H']:
        S[w]['V4.1 '+k]=g(V[k+'4']); S[w]['合併 '+k]=[mix([p,q],[.6,.4]) for p,q in zip(x,S[w]['V4.1 '+k])]
for pre in ['V4.1 ','合併 ']:
    print(f'== {pre.strip()}：年化/回撤/Calmar/Sharpe（05-15 ｜ 16-20 ｜ 21-26 ｜ 16-26）')
    for k in ['FIB','FIB_S','FIB_H']:
        o=[]
        for w,lo,hi in [('cpold','2005','2015'),('cpnew','2016','2020'),('cpnew','2021','2026'),('cpnew','2016','2026')]:
            L=[z.loc[lo:hi] for z in S[w][pre+k]]; o.append(f'{np.median([cagr(z) for z in L])*100:.1f}%/{np.median([mdd(z) for z in L])*100:.1f}%/{np.median([cal(z) for z in L]):.2f}/{np.median([sh(z) for z in L]):.2f}')
        line=f' {k:6s} '+' ｜ '.join(o)
        if k!='FIB':
            h=pd.concat(S['cpnew'][pre+'FIB'],axis=1).mean(axis=1); vv=pd.concat(S['cpnew'][pre+k],axis=1).mean(axis=1); line+=f' ｜ P {boot(h,vv):.2f}'
        print(line)
