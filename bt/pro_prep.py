import pandas as pd, numpy as np, pickle, os
from load import load_all
from engine import prep as old_prep
CACHE='/home/claude/bt/prep_pro.pkl' if os.environ.get('RAWSRC','yahoo')=='yahoo' else ('/home/claude/bt/prep_pro_fm_dl.pkl' if os.environ.get('DELIST')=='1' else '/home/claude/bt/prep_pro_fm.pkl')
def prep_pro():
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    P=old_prep()   # 取 cal/mkt/taiex/Cw/Lw/RCw/codes/shares/lst
    lst,t,cal0,S,_=load_all()
    feats=[]
    for code,a in S.items():
        C,Hh,L,V=a.C,a.H,a.L,a.V
        n=np.arange(len(a))
        e200=C.ewm(span=200,adjust=False).mean(); slope=(e200/e200.shift(10)-1)*100/10
        brk=C>C.shift(1).rolling(5).max()
        hh=Hh.rolling(250).max(); dist=C/hh-1
        pc=C.shift(1); tr=pd.concat([Hh-L,(Hh-pc).abs(),(L-pc).abs()],axis=1).max(axis=1)
        atrp=tr.ewm(alpha=1/14,adjust=False).mean()/C*100
        chg=C/pc-1
        to20=(a.RC*V).rolling(20).mean()
        vr=V/V.rolling(20).mean(); novol=vr.shift(1).rolling(5).max()<=3
        s=((atrp<=3.3).astype(int)+(dist>=-0.08).astype(int)+(to20<20e6).astype(int)+(a.RC<30).astype(int))
        pos_main=dist>=-0.12; rescue=(~pos_main)&(atrp<=2.5)
        base=(n>=250)&(slope>=0.1)&brk&(pos_main|rescue)&(chg<0.095)
        f=pd.DataFrame({'code':code,'score':s,'rc':a.RC,'atrp':atrp,'rescue':rescue,'base':base,'dist':dist,'chg':chg,
                        'to20':to20,'slope':slope,'slope150':slope,'novol':novol,
                        's_atr':atrp<=3.3,'s_dist':dist>=-0.08,'s_to':to20<20e6,'s_px':a.RC<30})
        feats.append(f[f.base])
    sig=pd.concat(feats).reset_index().rename(columns={'index':'Date'})
    sig=sig[sig.Date.isin(P['cal'])]; sig['mkt']=sig.Date.map(P['mkt'])
    Q=dict(P); Q['sig']=sig
    pickle.dump(Q,open(CACHE,'wb')); return Q
