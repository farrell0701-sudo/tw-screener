import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
X=pd.read_pickle('/home/claude/old/L098X_cpnew.pkl'); V=pd.read_pickle('/home/claude/old/L098V_cpnew.pkl'); a,b=X['_window']
g=lambda L:[pd.Series(s).loc['2016-01-01':b].dropna() for s in L]
xb,xq,vb,vq=g(X['BASE']),g(X['Q']),g(V['FIB4']),g(V['FIBQ4'])
S={'X1 基準':xb,'X1 品質':xq,'V4.1 基準':vb,'V4.1 品質':vq,
   '合併 基準':[mix([x,v],[.6,.4]) for x,v in zip(xb,vb)],'合併 QX':[mix([x,v],[.6,.4]) for x,v in zip(xq,vb)],
   '合併 QV':[mix([x,v],[.6,.4]) for x,v in zip(xb,vq)],'合併 QB':[mix([x,v],[.6,.4]) for x,v in zip(xq,vq)]}
def row(k):
    o=[]
    for nm,lo,hi in [('16-20','2016','2020'),('21-26','2021','2026'),('16-26','2016','2026')]:
        P=[x.loc[lo:hi] for x in S[k]]; o.append(f'{nm} {np.median([cagr(x) for x in P])*100:.1f}%/{np.median([mdd(x) for x in P])*100:.1f}%/{np.median([cal(x) for x in P]):.2f}/夏普{np.median([sh(x) for x in P]):.2f}')
    return ' ｜ '.join(o)
for k in S: print(f'{k:8s}', row(k))
for base,k in [('X1 基準','X1 品質'),('V4.1 基準','V4.1 品質'),('合併 基準','合併 QX'),('合併 基準','合併 QV'),('合併 基準','合併 QB')]:
    h=pd.concat(S[base],axis=1).mean(axis=1); v=pd.concat(S[k],axis=1).mean(axis=1); print(f'{k} vs {base}: P {boot(h,v):.2f}')
