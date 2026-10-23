import pandas as pd, numpy as np, pickle, os
from load import load_all

CACHE='/home/claude/bt/prep2.pkl' if os.environ.get('RAWSRC','yahoo')=='yahoo' else ('/home/claude/bt/prep2_fm_dl.pkl' if os.environ.get('DELIST')=='1' else '/home/claude/bt/prep2_fm.pkl')

def prep():
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    lst,t,cal0,S,_=load_all()
    # 股本: 最新集保 level 17 總股數
    td=pd.read_csv('/home/claude/farrell0701-sudo/tw-screener/data/tdcc/20261008.csv',dtype={'code':str})
    shares=td[td.level==17].set_index('code').shares
    # 主行事曆: 有 >=40% 股票交易的日子
    cnt=pd.Series(0,dtype=int)
    allidx=pd.concat([pd.Series(1,index=a.index) for a in S.values()])
    cnt=allidx.groupby(level=0).size()
    cal=cnt.index[cnt>=0.4*cnt.rolling(60,min_periods=1).max()]
    cal=cal.sort_values()
    # 大盤濾網: TAIEX 自身K棒算 EMA60, 與 20 根前比較, 再 ffill 到主行事曆
    tc=t.Close
    e60=tc.ewm(span=60,adjust=False).mean()
    mkt=(e60/e60.shift(20)-1>0)
    mkt=mkt.reindex(cal,method='ffill').fillna(False)
    taiex=tc.reindex(cal,method='ffill')
    feats=[]
    for code,a in S.items():
        C,Hh,L,V,O=a.C,a.H,a.L,a.V,a.O
        n=np.arange(len(a))
        e200=C.ewm(span=200,adjust=False).mean()
        slope=(e200/e200.shift(10)-1)*100/10
        e150=C.ewm(span=150,adjust=False).mean()
        slope150=(e150/e150.shift(10)-1)*100/10
        brk=C>C.shift(1).rolling(5).max()
        volr=V/V.rolling(20).mean()
        novol=volr.shift(1).rolling(5).max()<=3
        hh=Hh.rolling(250).max()
        dist=C/hh-1
        pc=C.shift(1)
        tr=pd.concat([Hh-L,(Hh-pc).abs(),(L-pc).abs()],axis=1).max(axis=1)
        atr=tr.ewm(alpha=1/14,adjust=False).mean()
        atrp=atr/C*100
        chg=C/pc-1
        rng=(Hh.rolling(20).max()-L.rolling(20).min())/C*100
        e60=C.ewm(span=60,adjust=False).mean()
        d60=(C/e60-1)*100
        cap_ok=bool(shares.get(code,0)*10/1e8>=10)
        s=( (atrp<=3.3).astype(int)+(dist>=-0.08).astype(int)+(rng<=16.49).astype(int)
           +(d60<=11.90).astype(int)+(chg<=0.0268).astype(int)+int(cap_ok))
        pos_main=dist>=-0.12; pos_rescue=(~pos_main)&(atrp<=2.5)
        to20=(a.RC*V).rolling(20).mean()   # 20日平均成交金額(元)
        base=(n>=250)&((slope>=0.1)|(slope150>=0.1))&brk&novol&(pos_main|pos_rescue)&(chg<0.095)
        f=pd.DataFrame({'code':code,'score':s,'rc':a.RC,'atrp':atrp,'rescue':pos_rescue,
                        'base':base,'dist':dist,'chg':chg,'to20':to20,'slope':slope,'slope150':slope150})
        feats.append(f[f.base])
    sig=pd.concat(feats).reset_index().rename(columns={'index':'Date'})
    sig=sig[sig.Date.isin(cal)]
    sig['mkt']=sig.Date.map(mkt)
    codes=sorted(S)
    Cw=pd.DataFrame({c:S[c].C for c in codes}).reindex(cal)
    Lw=pd.DataFrame({c:S[c].L for c in codes}).reindex(cal)
    RCw=pd.DataFrame({c:S[c].RC for c in codes}).reindex(cal)
    out=dict(cal=cal,mkt=mkt,taiex=taiex,sig=sig,Cw=Cw,Lw=Lw,RCw=RCw,codes=codes,shares=shares,lst=lst)
    pickle.dump(out,open(CACHE,'wb'))
    return out

DEF=dict(start='2016-01-01',end='2026-12-31',cap0=1_000_000,slots=8,w=0.125,minscore=3,
         slip=0.001,fee=0.000225,tax=0.00405,atr_mult=4,dmin=0.04,dmax=0.20,be_trig=1.06,be_lvl=1.005,
         use_mkt=True,use_rescue=True,pos12=-0.12,limitdown_block=False,min_to=0,rank='score_price',seed=None,minfee=0,trend='ema200',exit='atr',trail=0.08,excl_ind=(),contrib=0,slip_tier=False,use_novol=False,max_part=0,ts_days=0,ts_ret=0.15)

def run(P,cfg=None):
    c=dict(DEF); c.update(cfg or {})
    cal=P['cal']; cal=cal[(cal>=c['start'])&(cal<=c['end'])]
    sig=P['sig']
    sig=sig[(sig.score>=c['minscore'])]
    sig=sig[(sig.slope if c['trend']=='ema200' else sig.slope150)>=0.1]
    if c['excl_ind']: sig=sig[~sig.code.map(P['lst'].set_index('code').industry).isin(c['excl_ind'])]
    if c['use_mkt']: sig=sig[sig.mkt]
    if not c['use_rescue']: sig=sig[~sig.rescue]
    if c['pos12']!=-0.12: sig=sig[(sig.dist>=c['pos12'])|(sig.rescue&c['use_rescue'])]
    if c['min_to']>0: sig=sig[sig.to20>=c['min_to']]
    if c['use_novol'] and 'novol' in sig: sig=sig[sig.novol]
    rng=np.random.default_rng(c['seed'])
    if c['rank']=='random': sig=sig.assign(r=rng.random(len(sig))).sort_values(['Date','r'])
    elif c['rank']=='score': sig=sig.sort_values(['Date','score','code'],ascending=[True,False,True])
    else: sig=sig.sort_values(['Date','score','rc'],ascending=[True,False,True])
    byday={d:g for d,g in sig.groupby('Date')}
    ci={k:i for i,k in enumerate(P['codes'])}
    Cw=P['Cw'].loc[cal].to_numpy(); Lw=P['Lw'].loc[cal].to_numpy(); RCw=P['RCw'].loc[cal].to_numpy()
    cash=c['cap0']; pos={}   # code -> dict
    last=np.full(Cw.shape[1],np.nan)
    eq=[]; trades=[]; nheld=[]; flows=[]
    for i,d in enumerate(cal):
        fl=0.0
        if c['contrib'] and i>0 and d.month!=cal[i-1].month:
            cash+=c['contrib']; fl=c['contrib']
        flows.append(fl)
        row=Cw[i]; ok=~np.isnan(row); last[ok]=row[ok]
        # 1) 出場
        for code in list(pos):
            p=pos[code]; j=p['j']
            if not ok[j] or i==p['i']: continue
            C=row[j]
            if c['limitdown_block'] and p['pending']==False:
                pass
            p['maxc']=max(p['maxc'],C)
            stop=p['maxc']*(1-p['dist'])
            reason=None
            if c['exit']=='trail':
                if C<p['maxc']*(1-c['trail']): reason='回落8%'
            else:
                if C<stop: reason='ATR停損'
                if p['maxc']>=p['ec']*c['be_trig'] and C<p['ec']*c['be_lvl']:
                    reason=reason or '保本'
                if c['ts_days'] and not reason and (i-p['i'])>=c['ts_days'] and C<p['ec']*(1+c['ts_ret']): reason='時間'
            if reason:
                if c['limitdown_block']:
                    prc=RCw[i-1,j] if i>0 else np.nan
                    # 前一日原始收盤可能缺: 往回找
                    k=i-1
                    while k>=0 and np.isnan(RCw[k,j]): k-=1
                    prc=RCw[k,j] if k>=0 else np.nan
                    if RCw[i,j]/prc-1<=-0.095 and abs(Lw[i,j]-C)<1e-9:
                        continue   # 跌停鎖死賣不掉, 明天再試
                px=C*(1-p['sl'])
                g=p['sh']*px
                proceeds=g-max(g*c['fee'],c['minfee'])-g*c['tax']
                cash+=proceeds
                trades.append(dict(code=code,entry=p['date'],exit=d,ep=p['ep'],xp=px,sh=p['sh'],
                    cost=p['cost'],proceeds=proceeds,ret=proceeds/p['cost']-1,days=i-p['i'],
                    reason=reason,score=p['score'],rescue=p['rescue'],atrp=p['atrp']))
                del pos[code]
        # 2) 進場
        mv=sum(p['sh']*last[p['j']] for p in pos.values())
        equity=cash+mv
        free=c['slots']-len(pos)
        if free>0 and d in byday:
            for r in byday[d].itertuples():
                if free<=0: break
                if r.code in pos: continue
                j=ci[r.code]; C=row[j]
                target=equity*c['w']
                amt=target if cash>=target else cash*0.99
                if c['max_part']>0: amt=min(amt,r.to20*c['max_part'])
                sl=c['slip']
                if c['slip_tier']:
                    sl=0.005 if r.to20<3e6 else (0.003 if r.to20<10e6 else 0.001)
                px=C*(1+sl)
                rawpx=RCw[i,j]*(1+sl)
                rsh=np.floor(amt/(rawpx*(1+c['fee'])))
                while rsh>=1 and rsh*rawpx+max(rsh*rawpx*c['fee'],c['minfee'])>amt: rsh-=1
                if rsh<1: continue
                gross=rsh*rawpx
                cost=gross+max(gross*c['fee'],c['minfee'])
                sh=gross/px   # 換成還原價單位的股數
                cash-=cost
                dist=min(max(r.atrp/100*c['atr_mult'],c['dmin']),c['dmax'])
                pos[r.code]=dict(sl=sl,j=j,i=i,date=d,ec=C,ep=px,sh=sh,cost=cost,maxc=C,dist=dist,
                                 score=r.score,rescue=r.rescue,atrp=r.atrp,pending=False)
                free-=1
        mv=sum(p['sh']*last[p['j']] for p in pos.values())
        eq.append(cash+mv); nheld.append(len(pos))
    E=pd.Series(eq,index=cal)
    E.attrs['flows']=pd.Series(flows,index=cal)
    T=pd.DataFrame(trades)
    openpos=[dict(code=k,entry=v['date'],ep=v['ep'],last=last[v['j']]) for k,v in pos.items()]
    return E,T,pd.Series(nheld,index=cal),openpos

def stats(E,T=None,bench=None):
    r=E.pct_change().dropna()
    yrs=(E.index[-1]-E.index[0]).days/365.25
    cagr=(E.iloc[-1]/E.iloc[0])**(1/yrs)-1
    vol=r.std()*np.sqrt(252)
    sharpe=r.mean()/r.std()*np.sqrt(252)
    dn=r[r<0].std()*np.sqrt(252); sortino=r.mean()*252/dn
    dd=E/E.cummax()-1; mdd=dd.min()
    # 最長回撤期
    under=dd<0; dur=0;mx=0
    for u in under: dur=dur+1 if u else 0; mx=max(mx,dur)
    o=dict(CAGR=cagr,Vol=vol,Sharpe=sharpe,Sortino=sortino,MDD=mdd,Calmar=cagr/-mdd,MaxDDdays=mx,Final=E.iloc[-1]/E.iloc[0])
    if T is not None and len(T):
        w=T[T.ret>0]; l=T[T.ret<=0]
        o.update(N=len(T),Win=len(w)/len(T),AvgWin=w.ret.mean(),AvgLoss=l.ret.mean(),
                 Payoff=w.ret.mean()/-l.ret.mean(),PF=(w.proceeds-w.cost).sum()/-(l.proceeds-l.cost).sum(),
                 AvgRet=T.ret.mean(),MedDays=T.days.median(),AvgDays=T.days.mean())
    return o
