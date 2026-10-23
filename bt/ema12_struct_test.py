import os; os.environ['RAWSRC']='finmind'; os.environ['DELIST']='1'
import numpy as np, pandas as pd
from load import load_all
import ema12_opt as M
lst,t,cal,S,_=load_all()
codes=sorted(S)
W=lambda k: pd.DataFrame({c:S[c][k] for c in codes}).reindex(cal)
C,H,L,V,RC=W('C'),W('H'),W('L'),W('V'),W('RC')
f=C/RC; rawH=H/f; rawL=L/f
ix=t.Close
cat_map=lst.set_index('code').industry.to_dict()
P=M.build_panel(C,H,L,RC*V,RC,rawH,rawL,V,ix,cat_map)
# 結構特徵（逐檔、只用有交易的 K 棒）
ov,hv={}, {}
for c in codes:
    a=S[c]; h,l,cl=a.H,a.L,a.C
    hA,lA=h.rolling(10).max(),l.rolling(10).min(); hB,lB=h.shift(10).rolling(10).max(),l.shift(10).rolling(10).min()
    ov[c]=((np.minimum(hA,hB)-np.maximum(lA,lB)).clip(lower=0)/(hA-lA))
    w20=(h.rolling(20).max()-l.rolling(20).min())/cl
    hv[c]=(hA<hB)&(hB<h.shift(20).rolling(10).max())&(w20>w20.rolling(250,min_periods=60).quantile(0.75))&(cl/cl.shift(60)-1>0.3)
P['OVAB']=pd.DataFrame(ov).reindex(index=cal,columns=codes).values.astype(np.float32)
P['c_nohv']=~pd.DataFrame(hv).reindex(index=cal,columns=codes).fillna(False).values.astype(bool)
print('高位震盪比例',1-P['c_nohv'].mean())
M.TRAIN=('2016-03-01','2020-12-31'); M.TEST=('2021-01-01','2099-12-31'); M.N_STARTS=5
R=[]
B=M.BASE_ORIG; I=M.IMPROVED
M.both(P,B,'原始規則',R)
M.both(P,dict(B,extra=('c_nohv',)),'原始 + 排除高位震盪',R)
M.both(P,dict(B,rank='ld_ov'),'原始、排序 LD − 重疊度',R)
M.both(P,dict(B,rank='ld_ov',extra=('c_nohv',)),'原始 + 兩者',R)
M.both(P,I,'改良基準',R)
M.both(P,dict(I,extra=('c_nohv',)),'改良 + 排除高位震盪',R)
for pen in (0.5,1.0,2.0):
    M.both(P,dict(I,rank='ld_ov',ov_pen=pen),f'改良、LD − {pen}×重疊度',R)
M.both(P,dict(I,rank='ld_ov',extra=('c_nohv',)),'改良 + 兩者',R)
M.both(P,dict(I,mkt='slope60'),'改良 + 大盤濾網',R)
M.both(P,dict(I,mkt='slope60',rank='ld_ov',extra=('c_nohv',)),'改良 + 大盤濾網 + 兩者',R)
pd.DataFrame(R).to_csv('ema12_struct.csv',index=False)
