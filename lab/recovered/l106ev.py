import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
S={}; KV=None; KX=None
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    V=pd.read_pickle(f'/home/claude/old/L106V_{w}.pkl'); X=pd.read_pickle(f'/home/claude/old/L106X_{w}.pkl'); a,b=V['_window']
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]
    KV=[k for k in V if k!='_window']; KX=[k for k in X if k!='_window']
    vb=g(V['BASE']); xb=g(X['BASE']); S[w]={}
    for k in KV: vv=g(V[k]); S[w][('V',k)]=vv; S[w][('M',k)]=[mix([p,q],[.6,.4]) for p,q in zip(xb,vv)]
    for k in KX:
        xx=g(X[k]); S[w][('X',k)]=xx
        if k!='BASE': S[w][('M',k)]=[mix([p,q],[.6,.4]) for p,q in zip(xx,vb)]
c=lambda w,key,lo,hi: np.median([cal(z.loc[lo:hi]) for z in S[w][key]])
def row(pre,k,base):
    key=(pre,k); L=S['cpnew'][key]; O=S['cpold'][key]
    v=[c('cpold',key,'2005','2015'),c('cpnew',key,'2016','2020'),c('cpnew',key,'2021','2026'),c('cpnew',key,'2016','2026')]
    dd=np.median([mdd(z) for z in O]); line=f' {k:10s} {v[0]:.2f} ｜ {v[1]:.2f} ｜ {v[2]:.2f} ｜ {v[3]:.2f}（16-26 {np.median([cagr(z) for z in L])*100:.1f}%/{np.median([mdd(z) for z in L])*100:.1f}%/夏普{np.median([sh(z) for z in L]):.2f}；05-15 回撤 {dd*100:.1f}%）'
    if base is not None:
        bk=(pre,'BASE'); h=pd.concat(S['cpnew'][bk],axis=1).mean(axis=1); vv=pd.concat(L,axis=1).mean(axis=1); P=boot(h,vv)
        b=[c('cpold',bk,'2005','2015'),c('cpnew',bk,'2016','2020'),c('cpnew',bk,'2021','2026')]; bdd=np.median([mdd(z) for z in S['cpold'][bk]])
        ok=all(v[i]>=b[i]-0.05 for i in range(3)) and dd>=bdd-0.02 and P>=0.5
        line+=f' ｜ P {P:.2f}'+(' ｜ ★可拿掉' if (ok and pre=="M") else '')
    return line
for pre,lab,K in [('V','V4.1 單獨',KV),('M','合併版（V4.1 減法）',KV),('X','X1 單獨',KX),('M','合併版（X1 減法）',KX)]:
    print(f'== {lab}：Calmar 05-15 ｜ 16-20 ｜ 21-26 ｜ 16-26')
    for k in K: print(row(pre,k,None if k=='BASE' else 1))
