import os; os.environ['RAWSRC']='finmind'; os.environ['DELIST']='1'
import numpy as np, pandas as pd
from load import load_all
lst,t,cal,S,_=load_all()
TR,ACT,FL,CH,MLD,AMT=3.707,3.16,1.404,2.0,4.63,4.646e6
rows=[]
for code,a in S.items():
    H,L,C,V,RC=a.H,a.L,a.C,a.V,a.RC
    if len(C)<260: continue
    e=lambda n: C.ewm(span=n,adjust=False).mean()
    e12,e144,e169,e200=e(12),e(144),e(169),e(200)
    pc=C.shift(); tr=pd.concat([H-L,(H-pc).abs(),(L-pc).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False).mean()
    gap=e12-e144; ld=(e12-e200)/atr
    sc=((e12>e12.shift(3)).astype(int)+(L.rolling(9).min()>=e144).astype(int)+(gap>gap.shift(5)).astype(int)+(ld<MLD).astype(int))
    amt20=(RC*V).rolling(20).mean()
    ent=((C>e12)&(pc<=e12.shift())&(e200>e200.shift())&(e12>e144)&(e144>e169)&(e169>e200)&((C-e12)/atr<=CH)
         &(np.arange(len(C))>=200)&(sc>=2)&(amt20>AMT))
    ent=ent[ent.index>='2016-01-01']
    if not ent.any(): continue
    hA,lA=H.rolling(10).max(),L.rolling(10).min(); hB,lB=H.shift(10).rolling(10).max(),L.shift(10).rolling(10).min()
    hC,lC=H.shift(20).rolling(40).max(),L.shift(20).rolling(40).min(); wA=hA-lA
    inter=lambda h,l:(np.minimum(hA,h)-np.maximum(lA,l)).clip(lower=0)
    F=pd.DataFrame({'W10':wA/C,'W20':(H.rolling(20).max()-L.rolling(20).min())/C,'W60':(H.rolling(60).max()-L.rolling(60).min())/C,
        'ovAB':inter(hB,lB)/wA,'ovAC':inter(hC,lC)/wA,'up':(hA+lA)/2>(hB+lB)/2,'newhi':H>=H.rolling(60).max(),
        'lowerhi':(hA<hB)&(hB<H.shift(20).rolling(10).max()),'g60':C/C.shift(60)-1,'ld':ld})
    F['W10_60']=F.W10/F.W60
    Cv,Hv,Av=C.values,H.values,atr.values; pos={d:i for i,d in enumerate(C.index)}
    for d in ent.index[ent.values]:
        i=pos[d]
        if i+1>=len(Cv): continue
        ep,ea,hh=Cv[i],Av[i],Cv[i]; fill=Cv[i+1]; out=None
        for k in range(i+1,len(Cv)-1):
            hh=max(hh,Hv[k]); stop=hh-TR*ea
            if hh>=ep+ACT*ea: stop=max(stop,ep+FL*ea)
            if Cv[k]<stop: out=Cv[k+1]; break
        if out is None: continue
        ret=out*0.999*(1-0.000225-0.00405)/(fill*1.001*1.000225)-1
        rows.append(dict(code=code,Date=d,ret=ret,days=k+1-(i+1),**F.loc[d].to_dict()))
R=pd.DataFrame(rows); R.to_pickle('overlap_ema12.pkl'); print('訊號數',len(R))
