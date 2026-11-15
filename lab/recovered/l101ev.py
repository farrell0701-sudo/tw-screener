import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    P=pd.read_pickle(f'/home/claude/old/L100P_{w}.pkl'); N=pd.read_pickle(f'/home/claude/old/L101_{w}.pkl'); a,b=P['_window']; V=pd.read_pickle(f'/home/claude/old/L096_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]; v=g(V['FIB4'])
    X={'NONE':g(N['NONE']),'HOT':g(P['BASE']),'SLOPE':g(N['SLOPE']),'BOTH':g(P['W60q20'])}
    S[w]={**{'X1 '+k:x for k,x in X.items()},**{'合併 '+k:[mix([p,q],[.6,.4]) for p,q in zip(x,v)] for k,x in X.items()}}
for pre in ['X1 ','合併 ']:
    print(f'== {pre.strip()}：年化/回撤/Calmar/Sharpe（05-15 ｜ 16-20 ｜ 21-26 ｜ 16-26）')
    for k in ['NONE','HOT','SLOPE','BOTH']:
        o=[]
        for w,lo,hi in [('cpold','2005','2015'),('cpnew','2016','2020'),('cpnew','2021','2026'),('cpnew','2016','2026')]:
            L=[x.loc[lo:hi] for x in S[w][pre+k]]; o.append(f'{np.median([cagr(x) for x in L])*100:.1f}%/{np.median([mdd(x) for x in L])*100:.1f}%/{np.median([cal(x) for x in L]):.2f}/{np.median([sh(x) for x in L]):.2f}')
        line=f' {k:5s} '+' ｜ '.join(o)
        if k!='NONE':
            h=pd.concat(S['cpnew'][pre+'NONE'],axis=1).mean(axis=1); vv=pd.concat(S['cpnew'][pre+k],axis=1).mean(axis=1); line+=f' ｜ vs NONE P {boot(h,vv):.2f}'
        print(line)
