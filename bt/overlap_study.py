import os; os.environ['RAWSRC']='finmind'; os.environ['DELIST']='1'
import numpy as np, pandas as pd
from load import load_all
from pro_prep import prep_pro
Q=prep_pro(); ind=Q['lst'].set_index('code').industry
sig=Q['sig']; sig=sig[(sig.score>=2)&sig.mkt&(sig.to20>=3e6)&~sig.code.map(ind).isin(['生技醫療業','通信網路業'])&(sig.Date>='2016-01-01')]
lst,t,cal,S,_=load_all()
rows=[]
for code,g in sig.groupby('code'):
    a=S[code]; H,L,C=a.H,a.L,a.C
    hA,lA=H.rolling(10).max(),L.rolling(10).min()
    hB,lB=H.shift(10).rolling(10).max(),L.shift(10).rolling(10).min()
    hC,lC=H.shift(20).rolling(40).max(),L.shift(20).rolling(40).min()
    wA=(hA-lA); inter=lambda h,l:(np.minimum(hA,h)-np.maximum(lA,l)).clip(lower=0)
    f=pd.DataFrame({
        'W10':wA/C,'W20':(H.rolling(20).max()-L.rolling(20).min())/C,'W60':(H.rolling(60).max()-L.rolling(60).min())/C,
        'ovAB':inter(hB,lB)/wA,'ovAC':inter(hC,lC)/wA,
        'up':(hA+lA)/2>(hB+lB)/2,'newhi':H>=H.rolling(60).max(),
        'lowerhi':(hA<hB)&(hB<H.shift(20).rolling(10).max()),
        'g60':C/C.shift(60)-1})
    f['W10_60']=f.W10/f.W60
    # 獨立交易（PRO 出場）
    Cv=C.values; idx=C.index; pos={d:i for i,d in enumerate(idx)}
    for r in g.itertuples():
        i=pos.get(r.Date)
        if i is None: continue
        ec=Cv[i]; d=min(max(r.atrp/100*4,0.04),0.15); mx=ec; out=None
        for k in range(i+1,len(Cv)):
            c=Cv[k]; mx=max(mx,c)
            if c<mx*(1-d) or (mx>=ec*1.06 and c<ec*1.005): out=c; break
        if out is None: continue
        sl=0.003 if r.to20<1e7 else 0.001
        ret=out*(1-sl)*(1-0.000225-0.00405)/(ec*(1+sl)*1.000225)-1
        fr=f.loc[r.Date]
        rows.append(dict(code=code,Date=r.Date,ret=ret,days=k-i,fwd20=Cv[min(i+20,len(Cv)-1)]/ec-1,**fr.to_dict()))
R=pd.DataFrame(rows); R.to_pickle('overlap_events.pkl'); print('訊號數',len(R))
