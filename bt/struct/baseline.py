import os, sys
os.environ['RAWSRC']='finmind'; os.environ['DELIST']='1'; sys.path.insert(0,'/home/claude/bt')
import numpy as np, pandas as pd
from load import load_all
lst,tx,cal,S,_=load_all(); rng=np.random.default_rng(0)
def label(H,L,C,t):
    aL=L[t-4:t+1].min(); ent=C[t+1]
    for k in range(t+1,min(len(C),t+61)):
        if H[k]>=ent*1.15: return 'win'
        if C[k]<aL*0.99: return 'loss'
    return 'none'
rows=[]
for code,a in S.items():
    H,L,C,V,RC=(a[k].values.astype(float) for k in ('H','L','C','V','RC')); n=len(C)
    if n<300: continue
    Cs=pd.Series(C); e=lambda s: Cs.ewm(span=s,adjust=False).mean().values
    e12,e20,e144,e169,e200=e(12),e(20),e(144),e(169),e(200)
    pc=np.r_[np.nan,C[:-1]]; atr=pd.Series(np.nanmax(np.c_[H-L,np.abs(H-pc),np.abs(L-pc)],axis=1)).ewm(alpha=1/14,adjust=False).mean().values
    amt=pd.Series(RC*V).rolling(20).mean().values
    align=(e12>e20)&(e20>e144)&(e144>e169)&(e169>e200)
    gap=e12-e144; ld=(e12-e200)/atr
    sc=((e12>np.r_[[np.nan]*3,e12[:-3]]).astype(int)+(pd.Series(L).rolling(9).min().values>=e144)+(gap>np.r_[[np.nan]*5,gap[:-5]])+(ld<4.63))
    ema_sig=(C>e12)&(pc<=np.r_[np.nan,e12[:-1]])&(e200>np.r_[np.nan,e200[:-1]])&(e12>e144)&(e144>e169)&(e169>e200)&((C-e12)/atr<=2)&(sc>=2)&(amt>4.646e6)
    ok=np.arange(n); ok=(ok>=260)&(ok+61<n)
    for t in np.where(ema_sig&ok)[0]:
        rows.append(('EMA12 訊號',a.index[t],label(H,L,C,t),C[min(n-1,t+61)]/C[t+1]-1))
    al=np.where(align&ok&(amt>3e6))[0]
    for t in rng.choice(al,min(len(al),15),replace=False) if len(al) else []:
        rows.append(('隨機多頭排列日',a.index[t],label(H,L,C,t),C[min(n-1,t+61)]/C[t+1]-1))
B=pd.DataFrame(rows,columns=['group','date','label','fwd60']); B.to_pickle('/home/claude/bt/struct/baseline.pkl')
E=pd.read_pickle('/home/claude/bt/struct/events.pkl'); E=E[E.complete]
E2=pd.DataFrame({'group':'結構轉變事件','date':E.date,'label':E.label,'fwd60':E.fwd60})
A=pd.concat([B,E2]); A['期']=np.where(A.date<'2021-01-01','16-20','21-26')
g=A.groupby(['group','期']).agg(n=('label','size'),成功=('label',lambda s:(s=='win').mean()),失敗=('label',lambda s:(s=='loss').mean()),fwd60平均=('fwd60','mean'),fwd60中位=('fwd60','median'))
print(g.round(3).to_string())
