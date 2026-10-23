import pandas as pd, numpy as np
t=pd.read_csv('/home/claude/hdata/TAIEX.csv',parse_dates=['Date']).set_index('Date'); t=t[t.index<'2026-10-16']
C,H,L,V=t.Close,t.High,t.Low,t.Volume
def ind(trendLen=200):
    e=C.ewm(span=trendLen,adjust=False).mean(); slope=(e/e.shift(10)-1)*100/10
    e60=C.ewm(span=60,adjust=False).mean(); mkt=e60/e60.shift(20)-1>0
    brk=C>C.shift(1).rolling(5).max()
    vr=V/V.rolling(20).mean(); novol=vr.shift(1).rolling(5).max()<=3
    hh=H.rolling(250).max(); dist=C/hh-1
    pc=C.shift(1); tr=pd.concat([H-L,(H-pc).abs(),(L-pc).abs()],axis=1).max(axis=1); atrp=tr.ewm(alpha=1/14,adjust=False).mean()/C*100
    chg=C/pc-1; rng=(H.rolling(20).max()-L.rolling(20).min())/C*100; d60=(C/e60-1)*100
    sc=(atrp<=3.3).astype(int)+(dist>=-0.08)+(rng<=16.49)+(d60<=11.9)+(chg<=0.0268)+1  # 股本項視為成立
    pos=(dist>=-0.12)|(atrp<=2.5)
    return dict(slope=slope,mkt=mkt,brk=brk,novol=novol,pos=pos,chg=chg,sc=sc,atrp=atrp)
def bt(cfg):
    c=dict(trend=200,slopemin=0.1,use_mkt=True,use_vol=True,minscore=3,mult=4,dmin=.04,dmax=.2,be=1.06,bel=1.005,cost=0.0002,start='2016-01-01',exit='atr',trail=0.08); c.update(cfg)
    I=ind(c['trend'])
    sig=(I['slope']>=c['slopemin'])&I['brk']&I['pos']&(I['chg']<0.095)&(I['sc']>=c['minscore'])
    if c['use_mkt']: sig&=I['mkt']
    if c['use_vol']: sig&=I['novol']
    sig&=(np.arange(len(C))>=249)
    idx=C.index; s=idx.get_indexer([idx[idx>=c['start']][0]])[0]
    inpos=False; tr=[]; eq=[1.0]; cur=1.0; dlist=[idx[s-1]]
    for i in range(s,len(C)):
        x=C.iloc[i]
        if inpos:
            cur*=x/C.iloc[i-1]
            mx=max(mx,x); out=None
            if c['exit']=='trail':
                if x<mx*(1-c['trail']): out='trail'
            else:
                if x<mx*(1-d): out='ATR'
                elif mx>=ec*c['be'] and x<ec*c['bel']: out='BE'
            if out:
                cur*=(1-c['cost']); tr.append(dict(e=idx[ei],x=idx[i],r=x/ec*(1-2*c['cost'])-1,days=i-ei,why=out)); inpos=False
        if not inpos and sig.iloc[i]:
            inpos=True; ec=x; mx=x; ei=i; d=min(max(I['atrp'].iloc[i]/100*c['mult'],c['dmin']),c['dmax']); cur*=(1-c['cost'])
        eq.append(cur); dlist.append(idx[i])
    E=pd.Series(eq,index=dlist); T=pd.DataFrame(tr)
    yrs=(E.index[-1]-E.index[0]).days/365.25; cagr=E.iloc[-1]**(1/yrs)-1; mdd=(E/E.cummax()-1).min()
    r=E.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(252)
    expo=None
    return dict(CAGR=cagr,MDD=mdd,Sh=sh,N=len(T),Win=(T.r>0).mean() if len(T) else 0,Avg=T.r.mean() if len(T) else 0,Days=T.days.mean() if len(T) else 0),E,T
bh=C[C.index>='2015-12-31']; yrs=(bh.index[-1]-bh.index[0]).days/365.25
print('買進持有 TAIEX: CAGR %.1f%% MDD %.1f%%'%(((bh.iloc[-1]/bh.iloc[0])**(1/yrs)-1)*100,((bh/bh.cummax()-1).min())*100))
for cfg,l in [({},'原規則'),({'use_vol':False},'不用量能條件'),({'minscore':0},'不用評分'),({'slopemin':0.05},'斜率≥0.05'),({'trend':100},'EMA100'),
              ({'mult':6},'ATR×6'),({'mult':8,'dmax':0.25},'ATR×8'),({'exit':'trail','trail':0.08},'回落8%出場'),({'exit':'trail','trail':0.05},'回落5%出場'),
              ({'use_mkt':False},'不用大盤濾網'),({'start':'2021-01-01'},'2021起')]:
    s,E,T=bt(cfg)
    expo=None
    print(f"{l:<14} CAGR {s['CAGR']*100:5.1f}% MDD {s['MDD']*100:6.1f}% Sh {s['Sh']:.2f} N {s['N']} 勝率 {s['Win']*100:4.0f}% 平均 {s['Avg']*100:5.2f}% 持有 {s['Days']:.0f}天")
s,E,T=bt({}); print(T.to_string())
