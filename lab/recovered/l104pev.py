import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
K=['FIB','SB_W60q20','SB_W60q10','SB_W60q30','SB_W40q20','SB_W90q20']; S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    V=pd.read_pickle(f'/home/claude/old/L102_{w}.pkl'); B=pd.read_pickle(f'/home/claude/old/L104_{w}.pkl'); P=pd.read_pickle(f'/home/claude/old/L104P_{w}.pkl'); a,b=V['_window']
    X=pd.read_pickle(f'/home/claude/old/L100P_{w}.pkl'); g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]; x=g(X['W60q20'])
    vv={'FIB':g(V['FIB4']),'SB_W60q20':g(B['FIB_SB4']),**{k:g(P[k+'4']) for k in K[2:]}}
    S[w]={**{('V',k):v for k,v in vv.items()},**{('M',k):[mix([p,q],[.6,.4]) for p,q in zip(x,v)] for k,v in vv.items()}}
for pre,lab in [('V','V4.1 單獨'),('M','小資金合併 60/40')]:
    print(f'== {lab}：Calmar 05-15 ｜ 16-20 ｜ 21-26 ｜ 16-26（16-26 年化/回撤/Sharpe；05-15 年化/回撤）｜ P')
    for k in K:
        c=lambda w,lo,hi: np.median([cal(z.loc[lo:hi]) for z in S[w][(pre,k)]])
        L=S['cpnew'][(pre,k)]; O=S['cpold'][(pre,k)]
        line=f' {k:10s} {c("cpold","2005","2015"):.2f} ｜ {c("cpnew","2016","2020"):.2f} ｜ {c("cpnew","2021","2026"):.2f} ｜ {c("cpnew","2016","2026"):.2f}（{np.median([cagr(z) for z in L])*100:.1f}%/{np.median([mdd(z) for z in L])*100:.1f}%/{np.median([sh(z) for z in L]):.2f}；{np.median([cagr(z) for z in O])*100:.1f}%/{np.median([mdd(z) for z in O])*100:.1f}%）'
        if k!='FIB':
            h=pd.concat(S['cpnew'][(pre,'FIB')],axis=1).mean(axis=1); v=pd.concat(L,axis=1).mean(axis=1); line+=f' ｜ P {boot(h,v):.2f}'
        print(line)
