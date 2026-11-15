import pandas as pd, numpy as np
exec(open('/home/claude/llab/l058ev.py').read().split("def worst20")[0])
S={}
for w,cut in [('cpold',lambda s:s.loc[:'2015-12-31']),('cpnew',lambda s:s.loc['2016-01-01':])]:
    E=pd.read_pickle(f'/home/claude/old/L096_{w}.pkl'); a,b=E['_window']; X=pd.read_pickle(f'/home/claude/old/L083X_{w}.pkl')
    g=lambda L:[cut(pd.Series(s).loc[a:b]).dropna() for s in L]
    S[w]={k:g(E[k]) for k in ['BASE4','FIB4','BASE10','FIB10']}; S[w]['X6']=g(X['X6_EW'])
    S[w]['MIXB']=[mix([x,v],[.6,.4]) for x,v in zip(S[w]['X6'],S[w]['BASE4'])]
    S[w]['MIXF']=[mix([x,v],[.6,.4]) for x,v in zip(S[w]['X6'],S[w]['FIB4'])]
def row(k):
    out=[]
    for w,sg in [('cpold',[('05-15','2005','2015')]),('cpnew',[('16-20','2016','2020'),('21-26','2021','2026'),('16-26','2016','2026')])]:
        for nm,lo,hi in sg:
            P=[x.loc[lo:hi] for x in S[w][k]]
            out.append(f'{nm} {np.median([cagr(x) for x in P])*100:.1f}%/{np.median([mdd(x) for x in P])*100:.1f}%/{np.median([cal(x) for x in P]):.2f}/夏普{np.median([sh(x) for x in P]):.2f}')
    return ' ｜ '.join(out)
print('（年化/回撤/Calmar/Sharpe，100 路徑中位）')
for b,f,lab in [('BASE4','FIB4','V4.1 4檔'),('BASE10','FIB10','V4.1 10檔'),('MIXB','MIXF','小資金合併 X6 60%+V4 40%')]:
    print(f'== {lab}'); print(' 對照', row(b)); print(' 剔除', row(f))
    h=pd.concat(S['cpnew'][b],axis=1).mean(1); v=pd.concat(S['cpnew'][f],axis=1).mean(1)
    win=[np.mean([cal(y.loc[lo:hi])>cal(x.loc[lo:hi]) for x,y in zip(S['cpnew'][b],S['cpnew'][f])]) for lo,hi in [('2016','2020'),('2021','2026')]]
    print(f' 2016-26 bootstrap P {boot(h,v):.2f} ｜ 路徑勝率 Calmar 16-20 {win[0]*100:.0f}% 21-26 {win[1]*100:.0f}%')
