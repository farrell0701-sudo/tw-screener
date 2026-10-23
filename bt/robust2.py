from engine import *
import json, numpy as np, itertools
P=prep()
NEW=dict(min_to=30e6,excl_ind=('生技醫療業','通信網路業'))
def S(cfg,label,base=NEW):
    c=dict(base); c.update(cfg); E,T,N,_=run(P,c); s=stats(E,T)
    return dict(label=label,CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],N=s.get('N',0),Win=s.get('Win',0),PF=s.get('PF',0),Final=s['Final'])
out={}
# 演進路徑: v3D → 最新版
V3D=dict(trend='ema150',exit='trail',rank='score',min_to=0,excl_ind=())
steps=[(V3D,'v3D 原版（EMA150、回落 8% 出場、只依分數排序）'),
 (dict(V3D,exit='atr'),'+ 出場改 ATR×4 移動停損 + 保本 6%'),
 (dict(V3D,exit='atr',trend='ema200'),'+ 趨勢改 EMA200'),
 (dict(V3D,exit='atr',trend='ema200',rank='score_price'),'+ 同分選低價'),
 (dict(V3D,exit='atr',trend='ema200',rank='score_price',min_to=30e6),'+ 20 日均額 ≥ 3 千萬'),
 ({},'+ 排除生技、通訊 ＝ 最新版')]
out['steps']=[S(c,l) for c,l in steps]
# 濾網組合
grid=[]
for mt in [0,3e6,10e6,30e6,50e6]:
    for ex,el in [((),'不排除'),(('生技醫療業','通信網路業'),'排除生技通訊')]:
        d=S({'min_to':mt,'excl_ind':ex},f'{mt/1e4:.0f}萬|{el}'); d['mt']=mt; d['ex']=el; grid.append(d)
out['grid']=grid
sens=[S({},'最新版（基準）')]
for cfg,l in [({'slip':0.002},'滑價 0.2%/邊'),({'slip':0.003},'滑價 0.3%/邊'),({'minfee':20,'cap0':100000},'10 萬本金 + 最低手續費 20 元'),({'limitdown_block':True},'跌停鎖死賣不掉→隔日再賣'),
 ({'minscore':2},'評分 ≥2'),({'minscore':4},'評分 ≥4'),({'atr_mult':3},'停損 ATR×3'),({'atr_mult':5},'停損 ATR×5'),
 ({'be_trig':1.04},'保本觸發 4%'),({'be_trig':1.08},'保本觸發 8%'),({'pos12':-0.10},'位置 −10%'),({'use_rescue':False},'不用救回組'),({'use_mkt':False},'不用大盤濾網'),
 ({'slots':5,'w':0.2},'5 格 × 20%'),({'slots':10,'w':0.1},'10 格 × 10%'),({'slots':12,'w':1/12},'12 格')]:
    sens.append(S(cfg,l))
out['sens']=sens
per=[]
for a,b,l in [('2016-01-01','2020-12-31','2016–2020'),('2021-01-01','2026-12-31','2021–2026'),('2016-01-01','2022-12-31','2016–2022')]:
    per.append(S({'start':a,'end':b},l))
for y in [2017,2018,2019,2020,2022,2024]: per.append(S({'start':f'{y}-01-01'},f'{y} 起'))
out['periods']=per
E,T,N,_=run(P,NEW); T.to_csv('trades_new.csv',index=False); E.to_csv('eq_new.csv')
T['pnl']=T.proceeds-T.cost
top=T.groupby('code').pnl.sum().sort_values(ascending=False).index.tolist()
names=P['lst'].set_index('code').name
drop=[]
for k in [1,3,5,10]:
    P2=dict(P); P2['sig']=P['sig'][~P['sig'].code.isin(top[:k])]
    c=dict(NEW); E2,T2,_,_=run(P2,c); s=stats(E2,T2)
    drop.append(dict(label=f'移除前 {k} 檔（{"、".join(names.get(x,x) for x in top[:min(k,5)])}{"…" if k>5 else ""}）',CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],N=s['N'],Win=s['Win'],PF=s['PF'],Final=s['Final']))
out['drop']=drop
rs=[]
for sd in range(300):
    c=dict(NEW,rank='random',seed=sd); E2,T2,_,_=run(P,c); s=stats(E2,T2); rs.append([s['CAGR'],s['MDD'],s['Sharpe']])
rs=np.array(rs); out['random']=dict(cagr=rs[:,0].tolist(),mdd=rs[:,1].tolist(),sharpe=rs[:,2].tolist())
r=E.pct_change().dropna().to_numpy(); n=len(r); rng=np.random.default_rng(1); B=20; bc=[];bm=[]
for _ in range(3000):
    idx=np.concatenate([np.arange(s,s+B) for s in rng.integers(0,n-B,n//B+1)])[:n]
    e=np.cumprod(1+r[idx]); bc.append(e[-1]**(252/n)-1); bm.append((e/np.maximum.accumulate(e)-1).min())
out['boot']=dict(cagr=np.percentile(bc,[5,25,50,75,95]).tolist(),mdd=np.percentile(bm,[5,25,50,75,95]).tolist())
json.dump(out,open('robust2.json','w'),ensure_ascii=False,default=float)
for k in ['steps','grid','sens','periods','drop']:
    print('==',k)
    for d in out[k]: print(f"{d['label']:<40} {d['CAGR']*100:6.1f}% {d['MDD']*100:6.1f}% Sh {d['Sharpe']:.2f} N {d['N']} PF {d['PF']:.2f}")
print('random',np.percentile(rs[:,0],[5,50,95]).round(3),'MDD',np.percentile(rs[:,1],[5,50,95]).round(3),'Sh',np.percentile(rs[:,2],[5,50,95]).round(2),'pct of base',(rs[:,0]<sens[0]['CAGR']).mean())
print('boot',np.round(out['boot']['cagr'],3),np.round(out['boot']['mdd'],3))
