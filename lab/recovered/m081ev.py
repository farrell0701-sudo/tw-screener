import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
D=pd.concat([pd.read_pickle(f'/home/claude/old/M081_{w}.pkl') for w in ['cpold','cpnew']])
D['seg']=np.where(D.date.dt.year<=2020,'14-20','21-26')
def resid(a,X):
    X=np.column_stack([np.ones(len(a))]+X); b=np.linalg.lstsq(X,a,rcond=None)[0]; return a-X@b
out=[]
for (f,g,s,t),G in D.groupby(['fac','grp','seg','t']):
    if len(G)<50: continue
    xr=G.x.rank().values; er=G.ex.rank().values; c=[G.r20.rank().values,G.r60.rank().values]
    ic=np.corrcoef(xr,er)[0,1]; pic=np.corrcoef(resid(xr,c),resid(er,c))[0,1]
    q=pd.qcut(G.x.rank(method='first'),10,labels=False)
    p10=G[q==9]; p1=G[q==0]
    vw=lambda H:np.average(H.ex,weights=H.mc)
    out.append(dict(fac=f,grp=g,seg=s,t=t,ic=ic,pic=pic,ls_ew=p10.ex.mean()-p1.ex.mean(),ls_vw=vw(p10)-vw(p1),p10=p10.ex.mean(),p10vw=vw(p10),top3=np.sort(p10.mc.values)[::-1][:3].sum()/p10.mc.sum()))
R=pd.DataFrame(out)
print('（每 21 日一期；IC 平均 / t / IC>0 比例 ｜ 偏 IC 平均 / t ｜ P10−P1 等權 / 市值加權 ｜ P10 對 U 超額 等權 / 市值加權 ｜ 市值加權 P10 前三大權重）')
for f in ['FNB5','FNB20','FPART20','TNB20']:
    print(f'== {f}')
    for g in ['全體','大型']:
        for s in ['14-20','21-26']:
            X=R[(R.fac==f)&(R.grp==g)&(R.seg==s)]; n=len(X); tt=lambda v:v.mean()/v.std()*np.sqrt(n)
            print(f' {g} {s} n{n} IC {X.ic.mean():+.3f}/t{tt(X.ic):+.1f}/{(X.ic>0).mean()*100:.0f}% ｜ 偏IC {X.pic.mean():+.3f}/t{tt(X.pic):+.1f} ｜ P10−P1 {X.ls_ew.mean()*100:+.2f}/{X.ls_vw.mean()*100:+.2f} ｜ P10 {X.p10.mean()*100:+.2f}/{X.p10vw.mean()*100:+.2f} ｜ 前三大 {X.top3.mean()*100:.0f}%')
