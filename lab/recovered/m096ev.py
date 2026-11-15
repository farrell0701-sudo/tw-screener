import pandas as pd, numpy as np
M=[pd.read_pickle(f'/home/claude/old/M096_{w}.pkl') for w in ['cpold','cpnew']]
OV=[m['ov'] for m in M]; D=pd.concat([m['rows'] for m in M]).sort_values('date').reset_index(drop=True)
print('== 重疊（強勢池內，排名相關／標記交集）')
for lab,o in zip(['05-15','16-26'],OV):
    print(f' {lab} S20 vs 20日漲幅 {o.c20r.mean():+.2f} ｜ S60 vs 20日漲幅 {o.c60r.mean():+.2f} ｜ S20 vs S200 {o.c20_200.mean():+.2f} ｜ S60 vs S200 {o.c60_200.mean():+.2f} ｜ S20 vs S60 {o.c20_60.mean():+.2f}')
    print(f'       S20 最陡占池 {o.n20.mean()*100:.0f}%，其中也過熱 {o.h20.mean()*100:.0f}%、也 S200 最陡 {o.s20.mean()*100:.0f}%；在現行候選中再標記 {o.c20.mean()*100:.0f}%')
    print(f'       S60 最陡占池 {o.n60.mean()*100:.0f}%，其中也過熱 {o.h60.mean()*100:.0f}%、也 S200 最陡 {o.s60.mean()*100:.0f}%；在現行候選中再標記 {o.c60.mean()*100:.0f}%')
D['seg']=np.where(D.date.dt.year<=2015,'05-15','16-26'); rf=0.01/12
D['MKTx']=D.MKT+0.0025-rf; D['SMB']=D.U-(D.MKT+0.0025); D['MOM']=D.POOL-D.U
def ols(y,X,L=2):
    X=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
    XtXi=np.linalg.inv(X.T@X); S=(X*e[:,None]).T@(X*e[:,None])
    for l in range(1,L+1):
        w=1-l/(L+1); G=(X[l:]*e[l:,None]).T@(X[:-l]*e[:-l,None]); S+=w*(G+G.T)
    return b,b/np.sqrt(np.diag(XtXi@S@XtXi))
ann=12
print('== 增量（現行 X1 候選內：非過熱 ∧ 非 S200 最陡）')
for s in ['05-15','16-26','全期']:
    Z=D if s=='全期' else D[D.seg==s]
    for nm in ['S20','S60']:
        z=Z.dropna(subset=[nm+'_f']); y=z[nm+'_f'].values-rf
        b,t=ols(y,z[['MKTx','SMB','MOM']].values); b1,t1=ols(y,z[['MKTx']].values)
        d=z[nm+'_k']-z[nm+'_f']; p=Z[nm+'_pic'].dropna()
        print(f' {s} {nm} 標記比例 {Z[nm+"_n"].mean()*100:.0f}% ｜ 標記組 β {b1[1]:.2f} 三因子 α {b[0]*ann*100:+.1f}%（t {t[0]:+.1f}）｜ 保留−標記 {d.mean()*ann*100:+.1f}%/年（t {d.mean()/d.std()*np.sqrt(len(d)):+.1f}）｜ 偏 IC（−斜率，控制 X1c）{p.mean():+.3f}（t {p.mean()/p.std()*np.sqrt(len(p)):+.1f}）')
