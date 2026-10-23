from engine import *
import json
P=prep()
E,T,N,op=run(P)
cal=E.index; tx=P['taiex'].loc[cal]
bs=stats(tx)
r=E.pct_change().dropna(); rb=tx.pct_change().dropna()
beta=np.cov(r,rb)[0,1]/rb.var(); corr=np.corrcoef(r,rb)[0,1]
# 年度
def yr(s):
    y=s.resample('YE').last(); y0=pd.concat([pd.Series([s.iloc[0]],index=[s.index[0]-pd.Timedelta(days=1)]),y]); return y0.pct_change().dropna()
ys=yr(E); yb=yr(tx)
ydd=[(E[str(y)]/E[str(y)].cummax()-1).min() for y in ys.index.year]
m=E.resample('ME').last(); m=pd.concat([pd.Series([E.iloc[0]],index=[E.index[0]]),m]).pct_change().dropna()
# 容量: 進場金額 / 20日均成交額
sig=P['sig'].set_index(['Date','code'])
T['to20']=[sig.to20.get((pd.Timestamp(a),c),np.nan) for a,c in zip(T.entry,T.code)]
T['part']=T.cost/T.to20
dd=E/E.cummax()-1
# 回撤期間前三
eps=[];i=0;v=dd.values;idx=dd.index
while i<len(v):
    if v[i]<0:
        j=i
        while j<len(v) and v[j]<0: j+=1
        k=i+np.argmin(v[i:j]); eps.append((idx[i-1 if i>0 else 0],idx[k],idx[j] if j<len(v) else None,v[k])); i=j
    else: i+=1
eps=sorted(eps,key=lambda x:x[3])[:5]
T['pnl']=T.proceeds-T.cost
out=dict(stats=stats(E,T),bench=bs,beta=beta,corr=corr,exposure=float((N/8).mean()),
 years=[dict(y=int(a.year),s=float(b),bm=float(yb.iloc[i]),dd=float(ydd[i]),n=int((T.exit.dt.year==a.year).sum())) for i,(a,b) in enumerate(ys.items())],
 monthly=[[d.strftime('%Y-%m'),float(v)] for d,v in m.items()],
 eq=[[d.strftime('%Y-%m-%d'),float(e),float(b)] for d,e,b in zip(cal,E/E.iloc[0],tx/tx.iloc[0])],
 dd=[[d.strftime('%Y-%m-%d'),float(x)] for d,x in dd.items()],
 ddeps=[dict(peak=str(a.date()),trough=str(b.date()),rec=str(c.date()) if c is not None else '未回復',depth=float(d)) for a,b,c,d in eps],
 part=dict(med=float(T.part.median()),p90=float(T.part.quantile(.9)),last50_med=float(T.part.tail(50).median()),max=float(T.part.max())),
 reasons=T.reason.value_counts().to_dict(),
 retdist=T.ret.tolist(), days=T.days.tolist(),
 byscore=T.groupby('score').ret.agg(['count','mean','median']).reset_index().to_dict('records'),
 top=T.sort_values('pnl',ascending=False).head(10)[['code','entry','exit','ret','days']].astype(str).to_dict('records'),
 worst=T.sort_values('ret').head(5)[['code','entry','exit','ret','days']].astype(str).to_dict('records'),
 openpos=[dict(code=o['code'],entry=str(o['entry'].date()),ret=float(o['last']/o['ep']-1)) for o in op],
 top10share=float(T.nlargest(10,'pnl').pnl.sum()/T.pnl.sum()),
 losestreak=int(max((sum(1 for _ in g) for k,g in __import__('itertools').groupby(T.sort_values('exit').ret<=0) if k),default=0)))
names=P['lst'].set_index('code').name
T['name']=T.code.map(names)
T.sort_values('entry').to_csv('trades_base.csv',index=False,encoding='utf-8-sig')
for d in out['top']+out['worst']+out['openpos']: d['name']=names.get(d['code'],'')
json.dump(out,open('extras.json','w'),ensure_ascii=False,default=float)
o=out; print({k:round(v,3) for k,v in o['stats'].items()}); print('bench',{k:round(v,3) for k,v in o['bench'].items()}); print('beta',round(beta,2),'corr',round(corr,2),'exposure',round(o['exposure'],2))
print([(d['y'],round(d['s'],3),round(d['bm'],3),round(d['dd'],3),d['n']) for d in o['years']]); print(o['ddeps']); print(o['part']); print(o['losestreak'],o['top10share'])
print(o['top']); print(o['worst']); print(o['openpos'])
