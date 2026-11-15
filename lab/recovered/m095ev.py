import pandas as pd, numpy as np
D=pd.concat([pd.read_pickle(f'/home/claude/old/M095_{w}.pkl') for w in ['cpold','cpnew']]).sort_values('date').reset_index(drop=True)
D['seg']=np.where(D.date.dt.year<=2015,'05-15','16-26'); rf=0.01/12*21/21
D['MKTx']=D.MKT+0.0025-rf; D['SMB']=D.U-(D.MKT+0.0025); D['MOM']=D.POOL-D.U
def ols(y,X,L=2):
    X=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
    XtXi=np.linalg.inv(X.T@X); S=(X*e[:,None]).T@(X*e[:,None])
    for l in range(1,L+1):
        w=1-l/(L+1); G=(X[l:]*e[l:,None]).T@(X[:-l]*e[:-l,None]); S+=w*(G+G.T)
    return b,b/np.sqrt(np.diag(XtXi@S@XtXi))
ann=252/21
def stats(col,Z):
    z=Z.dropna(subset=[col]); y=z[col].values-rf
    b1,t1=ols(y,z[['MKTx']].values); b3,t3=ols(y,z[['MKTx','SMB','MOM']].values)
    return f'β {b1[1]:.2f} ｜ CAPM α {b1[0]*ann*100:+.1f}%（t {t1[0]:+.1f}）｜ 三因子 α {b3[0]*ann*100:+.1f}%（t {t3[0]:+.1f}）'
def ic(col,Z):
    v=Z[col].dropna(); return f'IC {v.mean():+.3f}（t {v.mean()/v.std()*np.sqrt(len(v)):+.1f}，>0 {(v>0).mean()*100:.0f}%）'
RANK=['T_ld','ΔT_ld10','X1c','斜率平緩','淺回撤','價值','低波動','ER20','外資參與度']; EXCL=['過熱','最陡斜率','深回撤','個股獨跌','營收衰退']
for s in ['全期','05-15','16-26']:
    Z=D if s=='全期' else D[D.seg==s]
    print(f'\n######## {s}（{len(Z)} 期）｜ 參考：強勢池 {stats("POOL",Z)}')
    print('== 排序類（前 20% 組合）')
    for k in RANK: print(f' {k:6s} {ic(k+"_ic",Z)} ｜ {stats(k+"_p",Z)}')
    print('== 剔除類（IC 為「未被標記／方向變數」；保留組 vs 被標記組）')
    for k in EXCL:
        z=Z.dropna(subset=[k+'_f']); d=(z[k+'_p']-z[k+'_f'])
        print(f' {k:5s} {ic(k+"_ic",Z)} ｜ 標記比例 {Z[k+"_n"].mean()*100:.0f}% ｜ 保留組 {stats(k+"_p",Z)} ｜ 被標記組 {stats(k+"_f",Z)} ｜ 保留−標記 {d.mean()*ann*100:+.1f}%/年（t {d.mean()/d.std()*np.sqrt(len(d)):+.1f}）')
