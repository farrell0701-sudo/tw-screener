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
ovh,drop,lo=g('ovh').fillna(1),g('drop20'),g('lo_ratio')
M.TRAIN=('2016-03-01','2020-12-31'); M.TEST=('2021-01-01','2099-12-31'); M.N_STARTS=5
R=[]; I=M.IMPROVED; B=M.BASE_ORIG
M.both(P,I,'改良規則（對照）',R)
for o_,d_,l_ in [(0.05,-0.23,1.0),(0.02,-0.23,1.0),(0.10,-0.23,1.0),(0.20,-0.23,1.0),(0.05,-0.18,1.0),(0.05,-0.28,1.0),
                 (0.05,-0.23,0.95),(0.05,-0.23,1.05),(0.05,-0.23,1.10),(0.05,None,1.0),(None,-0.23,1.0),(0.05,-0.23,None)]:
    m=np.ones(ovh.shape,bool)
    if o_ is not None: m&=(ovh<=o_).values
    if d_ is not None: m&=(drop<=d_).values
    if l_ is not None: m&=(lo<=l_).values
    P['b_x']=m.astype(np.float32)
    lab=f"上方壓力≤{o_} 急跌≤{d_} 低點比≤{l_}".replace('None','（不用）')
    M.both(P,dict(I,rank='ld_bonus',bonus=(('b_x',10),)),lab,R)
for sd in range(5):
    M.both(P,dict(I,rank='random',seed=sd),f'改良、隨機排序 #{sd}',R)
pd.DataFrame(R).to_csv('/home/claude/bt/struct/ema12_sens.csv',index=False)
