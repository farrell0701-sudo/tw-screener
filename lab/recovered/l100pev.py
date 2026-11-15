import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
K=['BASE','W60q20','W60q10','W60q30','W40q20','W90q20']; S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    E=pd.read_pickle(f'/home/claude/old/L100P_{w}.pkl'); a,b=E['_window']; V=pd.read_pickle(f'/home/claude/old/L096_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]; v=g(V['FIB4'])
    S[w]={k:[mix([x,y],[.6,.4]) for x,y in zip(g(E[k]),v)] for k in K}
print('小資金合併版：Calmar 05-15 ｜ 16-20 ｜ 21-26 ｜ 16-26（年化/回撤/Sharpe）｜ P')
for k in K:
    c=lambda w,lo,hi: np.median([cal(x.loc[lo:hi]) for x in S[w][k]])
    P=[x for x in S['cpnew'][k]]
    line=f' {k:7s} {c("cpold","2005","2015"):.2f} ｜ {c("cpnew","2016","2020"):.2f} ｜ {c("cpnew","2021","2026"):.2f} ｜ {c("cpnew","2016","2026"):.2f}（{np.median([cagr(x) for x in P])*100:.1f}%/{np.median([mdd(x) for x in P])*100:.1f}%/{np.median([sh(x) for x in P]):.2f}）'
    if k!='BASE':
        h=pd.concat(S['cpnew']['BASE'],axis=1).mean(axis=1); vv=pd.concat(S['cpnew'][k],axis=1).mean(axis=1); line+=f' ｜ P {boot(h,vv):.2f}'
    print(line)
