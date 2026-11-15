import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    E=pd.read_pickle(f'/home/claude/old/L097_{w}.pkl'); a,b=E['_window']; V=pd.read_pickle(f'/home/claude/old/L096_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]
    S[w]={k:g(E[k]) for k in ['BASE','PY6','PY9']}; v=g(V['FIB4'])
    for k in ['BASE','PY6','PY9']: S[w]['M'+k]=[mix([x,y],[.6,.4]) for x,y in zip(S[w][k],v)]
def row(k):
    out=[]
    for w,sg in [('cpold',[('05-15','2005','2015')]),('cpnew',[('16-20','2016','2020'),('21-26','2021','2026'),('16-26','2016','2026')])]:
        for nm,lo,hi in sg:
            P=[x.loc[lo:hi] for x in S[w][k]]
            out.append(f'{nm} {np.median([cagr(x) for x in P])*100:.1f}%/{np.median([mdd(x) for x in P])*100:.1f}%/{np.median([cal(x) for x in P]):.2f}/夏普{np.median([sh(x) for x in P]):.2f}/最差路徑回撤{np.min([mdd(x) for x in P])*100:.0f}%')
    return ' ｜ '.join(out)
print('（年化/回撤/Calmar/Sharpe/最差路徑回撤，100 路徑中位）')
for pre,lab in [('','X1 單獨'),('M','小資金合併 60/40')]:
    print(f'== {lab}'); print(' 基準', row(pre+'BASE'))
    for k in ['PY6','PY9']:
        print(f' {k} ', row(pre+k))
        h=pd.concat(S['cpnew'][pre+'BASE'],axis=1).mean(axis=1); v=pd.concat(S['cpnew'][pre+k],axis=1).mean(axis=1)
        win=[np.mean([cal(y.loc[lo:hi])>cal(x.loc[lo:hi]) for x,y in zip(S['cpnew'][pre+'BASE'],S['cpnew'][pre+k])]) for lo,hi in [('2016','2020'),('2021','2026')]]
        print(f'      2016-26 bootstrap P {boot(h,v):.2f} ｜ 路徑勝率 16-20 {win[0]*100:.0f}% 21-26 {win[1]*100:.0f}%')
