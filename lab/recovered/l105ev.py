import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
K=['BASE','BATCH','NEW1','NEW2']; S={}
w20=lambda z:((1+z).rolling(20).apply(np.prod,raw=True)-1).min()
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    E=pd.read_pickle(f'/home/claude/old/L105_{w}.pkl'); a,b=E['_window']; V=pd.read_pickle(f'/home/claude/old/L102_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]; v=g(V['FIB4'])
    S[w]={**{('X',k):g(E[k]) for k in K},**{('M',k):[mix([p,q],[.6,.4]) for p,q in zip(g(E[k]),v)] for k in K}}
for pre,lab in [('X','X1 單獨'),('M','小資金合併')]:
    print(f'== {lab}：Calmar 05-15 ｜ 16-20 ｜ 21-26 ｜ 16-26（16-26 年化/回撤/Sharpe/最差路徑回撤/最差20日中位；05-15 回撤）｜ P')
    for k in K:
        c=lambda ww,lo,hi: np.median([cal(z.loc[lo:hi]) for z in S[ww][(pre,k)]]); L=S['cpnew'][(pre,k)]; O=S['cpold'][(pre,k)]
        line=f' {k:5s} {c("cpold","2005","2015"):.2f} ｜ {c("cpnew","2016","2020"):.2f} ｜ {c("cpnew","2021","2026"):.2f} ｜ {c("cpnew","2016","2026"):.2f}（{np.median([cagr(z) for z in L])*100:.1f}%/{np.median([mdd(z) for z in L])*100:.1f}%/{np.median([sh(z) for z in L]):.2f}/{np.min([mdd(z) for z in L])*100:.0f}%/{np.median([w20(z) for z in L])*100:.1f}%；{np.median([mdd(z) for z in O])*100:.1f}%）'
        if k!='BASE':
            h=pd.concat(S['cpnew'][(pre,'BASE')],axis=1).mean(axis=1); vv=pd.concat(L,axis=1).mean(axis=1); line+=f' ｜ P {boot(h,vv):.2f}'
        print(line)
