import os, sys
os.environ['RAWSRC']='finmind'; os.environ['DELIST']='1'; sys.path.insert(0,'/home/claude/bt')
import numpy as np, pandas as pd
from load import load_all
import ema12_opt as M
lst,t,cal,S,_=load_all(); codes=sorted(S)
Wd=lambda k: pd.DataFrame({c:S[c][k] for c in codes}).reindex(cal)
C,H,L,V,RC=Wd('C'),Wd('H'),Wd('L'),Wd('V'),Wd('RC'); f=C/RC
P=M.build_panel(C,H,L,RC*V,RC,H/f,L/f,V,t.Close,lst.set_index('code').industry.to_dict())
D=pd.read_pickle('/home/claude/bt/struct/daily.pkl'); g=lambda k: D[k].reindex(index=cal,columns=codes)
DROP=float(sys.argv[1]) if len(sys.argv)>1 else -0.23
three=(g('ovh').fillna(1)<=0.05)&(g('drop20')<=DROP)&(g('lo_ratio')<=1.0)
P['c_three']=three.values; P['c_lowovh']=(g('ovh').fillna(1)<=0.2).values
P['b_three']=three.values.astype(np.float32); P['b_lowlo']=(g('lo_ratio')<=1.0).values.astype(np.float32)
P['b_ovh0']=(g('ovh').fillna(1)<=0.05).values.astype(np.float32)
print('三條件成立的股票日比例',round(float(three.values.mean()),4))
M.TRAIN=('2016-03-01','2020-12-31'); M.TEST=('2021-01-01','2099-12-31'); M.N_STARTS=5
R=[]; B=M.BASE_ORIG; I=M.IMPROVED
for base,bn in ((B,'原始'),(I,'改良')):
    M.both(P,base,f'{bn}規則',R)
    M.both(P,dict(base,extra=('c_three',)),f'{bn} ＋只做三條件',R)
    M.both(P,dict(base,extra=('c_lowovh',)),f'{bn} ＋排除上方壓力高',R)
    M.both(P,dict(base,rank='ld_bonus',bonus=(('b_three',10),)),f'{bn} ＋三條件優先',R)
    M.both(P,dict(base,rank='ld_bonus',bonus=(('b_lowlo',10),)),f'{bn} ＋破前低優先',R)
    M.both(P,dict(base,rank='ld_bonus',bonus=(('b_lowlo',10),('b_ovh0',5))),f'{bn} ＋破前低優先＋低壓力次之',R)
M.both(P,dict(I,mkt='slope60',rank='ld_bonus',bonus=(('b_three',10),)),'改良＋大盤濾網＋三條件優先',R)
pd.DataFrame(R).to_csv('/home/claude/bt/struct/ema12_portfolio.csv',index=False)
