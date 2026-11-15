import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
cols_s=[str(c) for c in C0.columns]; ind=D['ind'].reindex(C0.columns).fillna('未知').values
Cv=C0.values.astype(np.float64); T_,N_=Cv.shape
IX=pd.Series(np.asarray(ixd,dtype=np.float64),index=dates)
r1=Cv[1:]/Cv[:-1]-1; r1[~U[:-1]]=np.nan; EW=pd.Series(np.concatenate([[1.0],np.cumprod(1+np.nan_to_num(np.nanmean(r1,1)))]),index=dates)
print('== 年度：加權指數 ｜ 可交易池等權 ｜ 強勢池等權（每日再平衡近似）')
rp=np.where(pool[:-1],r1,np.nan); PL=pd.Series(np.concatenate([[1.0],np.cumprod(1+np.nan_to_num(np.nanmean(rp,1)))]),index=dates)
for y in range(2016,2027):
    s=str(y); 
    try: print(f' {y} 指數 {IX.loc[s].iloc[-1]/IX.loc[:str(y-1)].iloc[-1]-1:+.1%} ｜ 等權 {EW.loc[s].iloc[-1]/EW.loc[:str(y-1)].iloc[-1]-1:+.1%} ｜ 強勢池 {PL.loc[s].iloc[-1]/PL.loc[:str(y-1)].iloc[-1]-1:+.1%}')
    except Exception as e: pass
T=pd.read_pickle('/home/claude/old/M080F_cpnew.pkl')['X']; T['ind']=ind[T.j.values]; T['code']=[cols_s[j] for j in T.j.values]; T['y']=T.date.dt.year
T['seg']=np.where(T.y<=2020,'16-20','21-26')
print('== X1 交易：各產業筆數占比、平均淨報酬、對總損益貢獻（2016-20 vs 2021-26）')
for s in ['16-20','21-26']:
    X=T[T.seg==s]; g=X.groupby('ind').agg(n=('net','size'),avg=('net','mean'),tot=('net','sum')).sort_values('tot',ascending=False)
    g['占比']=g.n/len(X); g['貢獻']=g.tot/X.net.sum()
    print(f' -- {s} 全部 {len(X)} 筆，平均 {X.net.mean()*100:+.2f}%')
    print('  前八（貢獻）：'+'；'.join(f'{i} {r.占比*100:.0f}%筆/{r.avg*100:+.1f}%/{r.貢獻*100:.0f}%' for i,r in g.head(8).iterrows()))
    print('  後五（貢獻）：'+'；'.join(f'{i} {r.占比*100:.0f}%筆/{r.avg*100:+.1f}%/{r.貢獻*100:.0f}%' for i,r in g.tail(5).iterrows()))
print('== 2016-20 前 15 大贏家（去重交易）')
X=T[T.seg=='16-20'].sort_values('net',ascending=False).head(15)
print('  '+'；'.join(f'{r.code}({r.ind}) {r.date:%Y-%m} {r.net*100:+.0f}%' for _,r in X.iterrows()))
print('== 2016-20 年度 X1 交易平均與勝率')
for y in range(2016,2021):
    X=T[T.y==y]; print(f' {y} n{len(X)} 平均 {X.net.mean()*100:+.2f}% 中位 {X.net.median()*100:+.2f}% 勝率 {(X.net>0).mean()*100:.0f}% 前10%占損益 {np.sort(X.net)[::-1][:max(1,len(X)//10)].sum()/X.net.sum()*100:.0f}%')
# 強勢池產業組成 vs 各產業池外表現
idx=np.where((dates>=pd.Timestamp('2016-01-01')))[0][::20]
comp={}
for t in idx:
    y=dates[t].year; c=pd.Series(ind[pool[t]]).value_counts(normalize=True); comp.setdefault(y,[]).append(c)
print('== 強勢池產業組成（年平均，前六）')
for y in sorted(comp):
    c=pd.concat(comp[y],axis=1).fillna(0).mean(1).sort_values(ascending=False).head(6)
    print(f' {y} '+'，'.join(f'{k} {v*100:.0f}%' for k,v in c.items()))
