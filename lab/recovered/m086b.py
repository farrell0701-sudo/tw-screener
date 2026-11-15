import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
cols_s=[str(c) for c in C0.columns]; Cv=C0.values.astype(np.float64)
r1=Cv[1:]/Cv[:-1]-1
def ann(M):
    rp=np.where(M[:-1],r1,np.nan); s=pd.Series(np.concatenate([[0.0],np.nan_to_num(np.nanmean(rp,1))]),index=dates); return s
A={'強勢池全部':ann(pool),'非過熱(X1候選)':ann(pool&~hot),'過熱(被剔除)':ann(pool&hot)}
TL=pd.DataFrame(np.asarray(Tr,dtype=np.float64)); lv=pd.DataFrame(np.where(pool,TL.values,np.nan)).rank(axis=1,pct=True).values; ch=pd.DataFrame(np.where(pool,(TL-TL.shift(10)).values,np.nan)).rank(axis=1,pct=True).values
SC=0.5*np.nan_to_num(lv)+0.5*np.nan_to_num(ch); SC[~(pool&~hot)]=-1
top=np.zeros_like(pool); o=np.argsort(-SC,1)[:,:6]
for t in range(len(dates)): top[t,o[t]]=SC[t,o[t]]>0
A['X1c 每日前6名(無停損)']=ann(top)
print('年度報酬（等權每日再平衡近似，次日報酬）')
for y in range(2016,2027):
    print(f' {y} '+' ｜ '.join(f'{k} {(1+v.loc[str(y)]).prod()-1:+.0%}' for k,v in A.items()))
T=pd.read_pickle('/home/claude/old/M080F_cpnew.pkl')['X']; T['code']=[cols_s[j] for j in T.j.values]
for c in ['2327','2492','3026','6173','2456','8039']:
    if c in cols_s:
        j=cols_s.index(c); m=(dates>=pd.Timestamp('2017-01-01'))&(dates<=pd.Timestamp('2018-07-31'))
        X=T[(T.code==c)&(T.date>=pd.Timestamp('2017-01-01'))&(T.date<=pd.Timestamp('2018-07-31'))]
        print(f' {c} 2017/1–2018/7 在池日 {pool[m,j].mean()*100:.0f}% 其中過熱 {(pool&hot)[m,j].sum()/max(pool[m,j].sum(),1)*100:.0f}% ｜ 漲幅 {Cv[m,j][-1]/Cv[m,j][0]-1:+.0%}（最高 {np.nanmax(Cv[m,j])/Cv[m,j][0]-1:+.0%}）｜ X1 交易 {len(X)} 筆 '+', '.join(f'{r.date:%y-%m} {r.net*100:+.0f}%' for _,r in X.iterrows()))
