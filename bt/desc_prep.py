import pandas as pd, numpy as np, pickle, os
from load import load_all
from engine import prep as old_prep
CACHE='/home/claude/bt/prep_desc.pkl'
def prep_desc():
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    P=old_prep(); lst,t,cal0,S,_=load_all(); feats=[]
    for code,a in S.items():
        C,Hh,L,V=a.C,a.H,a.L,a.V; n=np.arange(len(a))
        e=C.ewm(span=200,adjust=False).mean(); slope=(e/e.shift(10)-1)*100/10
        brk=C>C.shift(1).rolling(5).max(); dist=C/Hh.rolling(250).max()-1
        pc=C.shift(1); tr=pd.concat([Hh-L,(Hh-pc).abs(),(L-pc).abs()],axis=1).max(axis=1)
        atrp=tr.ewm(alpha=1/14,adjust=False).mean()/C*100; chg=C/pc-1; to20=(a.RC*V).rolling(20).mean()
        pos=(dist>=-0.12); rescue=(~pos)&(atrp<=2.5)
        base=(n>=250)&(slope>=0.1)&(pos|rescue)&(to20>=3e6)
        f=pd.DataFrame({'code':code,'rc':a.RC,'atrp':atrp,'rescue':rescue,'dist':dist,'chg':chg,'to20':to20,'slope':slope,'slope150':slope,
            'brk':brk,'s4':(atrp<=3.3).astype(int)+(dist>=-0.08).astype(int)+(to20<20e6).astype(int)+(a.RC<30).astype(int),
            's2':(atrp<=3.3).astype(int)+(to20<20e6).astype(int)})
        feats.append(f[base])
    sig=pd.concat(feats).reset_index().rename(columns={'index':'Date'}); sig=sig[sig.Date.isin(P['cal'])]; sig['mkt']=sig.Date.map(P['mkt'])
    Q=dict(P); Q['sig']=sig; pickle.dump(Q,open(CACHE,'wb')); return Q
