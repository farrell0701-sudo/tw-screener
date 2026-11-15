import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    V=pd.read_pickle(f'/home/claude/old/L102_{w}.pkl'); a,b=V['_window']; X=pd.read_pickle(f'/home/claude/old/L100P_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]; x=g(X['W60q20'])
    S[w]={(k,wx):[mix([p,q],[wx,1-wx]) for p,q in zip(x,g(V[k+'4']))] for k in ['FIB','FIB_S'] for wx in [.3,.4,.5,.6,.7]}
c=lambda w,key,lo,hi: np.median([cal(z.loc[lo:hi]) for z in S[w][key]])
print('X1權重 ｜ V4.1版 ｜ Calmar 05-15 / 16-20 / 21-26 / 16-26 ｜ 16-26 年化/回撤/Sharpe ｜ FIB_S vs 同比例 FIB 的 P')
for wx in [.3,.4,.5,.6,.7]:
    for k in ['FIB','FIB_S']:
        L=S['cpnew'][(k,wx)]; P=''
        if k=='FIB_S':
            h=pd.concat(S['cpnew'][('FIB',wx)],axis=1).mean(axis=1); v=pd.concat(L,axis=1).mean(axis=1); P=f'P {boot(h,v):.2f}'
        print(f' {int(wx*100)}% ｜ {k:5s} ｜ {c("cpold",(k,wx),"2005","2015"):.2f} / {c("cpnew",(k,wx),"2016","2020"):.2f} / {c("cpnew",(k,wx),"2021","2026"):.2f} / {c("cpnew",(k,wx),"2016","2026"):.2f} ｜ {np.median([cagr(z) for z in L])*100:.1f}%/{np.median([mdd(z) for z in L])*100:.1f}%/{np.median([sh(z) for z in L]):.2f} ｜ {P}')
