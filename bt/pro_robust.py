from pro_prep import *; from engine import run,stats
import json, numpy as np, itertools
Q=prep_pro(); PRO=dict(minscore=2,min_to=3e6,excl_ind=('生技醫療業','通信網路業'),dmax=0.15,slip_tier=True)
def S(cfg,label):
    c=dict(PRO); c.update(cfg); E,T,N,_=run(Q,c); s=stats(E,T)
    return dict(label=label,CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],N=s.get('N',0),Win=s.get('Win',0),PF=s.get('PF',0),Final=s['Final'])
out={}
out['sens']=[S({},'回測基準')]+[S(c,l) for c,l in [
 ({'slip_tier':False,'slip':0.003},'滑價一律每邊 0.3%'),({'slip_tier':False,'slip':0.005},'滑價一律每邊 0.5%'),
 ({'cap0':100000,'minfee':20},'初始 10 萬 + 最低手續費 20 元'),({'limitdown_block':True},'跌停無法賣出，順延至隔日'),
 ({'slots':5,'w':0.2},'最多持有 5 檔'),({'slots':10,'w':0.1},'最多持有 10 檔'),({'slots':12,'w':1/12},'最多持有 12 檔')]]
out['params']=[S({},'回測基準')]+[S(c,l) for c,l in [
 ({'atr_mult':3},'停損倍數 −25%'),({'atr_mult':5},'停損倍數 +25%'),({'dmax':0.20},'停損上限放寬'),
 ({'be_trig':1.04},'保本啟動點降低'),({'be_trig':1.08},'保本啟動點提高'),({'minscore':1},'評分門檻 −1'),({'minscore':3},'評分門檻 +1'),
 ({'pos12':-0.10},'位置門檻收緊'),({'use_rescue':False},'取消例外組'),({'use_novol':True},'加回量能過濾')]]
per=[]
for a,b,l in [('2016-01-01','2020-12-31','2016–2020'),('2021-01-01','2026-12-31','2021–2026'),('2016-01-01','2022-12-31','2016–2022')]:
    per.append(S({'start':a,'end':b},l))
for y in [2017,2018,2019,2020,2022,2024]: per.append(S({'start':f'{y}-01-01'},f'{y}-01 起始至今'))
out['periods']=per
cap=[]
for c0 in [1e6,3e6,5e6,1e7,3e7]:
    for mp in [0.05,0.10,0.20]:
        c=dict(PRO,cap0=c0,max_part=mp); E,T,N,_=run(Q,c); s=stats(E,T)
        cap.append(dict(cap0=c0,mp=mp,CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],final=float(E.iloc[-1])))
out['capacity']=cap
E,T,N,_=run(Q,PRO); T['pnl']=T.proceeds-T.cost
top=T.groupby('code').pnl.sum().sort_values(ascending=False).index.tolist(); names=Q['lst'].set_index('code').name
drop=[]
for k in [1,3,5,10]:
    Q2=dict(Q); Q2['sig']=Q['sig'][~Q['sig'].code.isin(top[:k])]
    E2,T2,_,_=run(Q2,PRO); s=stats(E2,T2)
    drop.append(dict(label=f'移除前 {k} 檔（{"、".join(names.get(x,x) for x in top[:min(k,5)])}{"…" if k>5 else ""}）',CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],N=s['N'],Win=s['Win'],PF=s['PF'],Final=s['Final']))
out['drop']=drop
rs=[]
for sd in range(300):
    E2,T2,_,_=run(Q,dict(PRO,rank='random',seed=sd)); s=stats(E2,T2); rs.append([s['CAGR'],s['MDD'],s['Sharpe']])
rs=np.array(rs); out['random']=dict(cagr=rs[:,0].tolist(),mdd=rs[:,1].tolist(),sharpe=rs[:,2].tolist())
r=E.pct_change().dropna().to_numpy(); n=len(r); rng=np.random.default_rng(1); B=20; bc=[];bm=[]
for _ in range(3000):
    idx=np.concatenate([np.arange(s,s+B) for s in rng.integers(0,n-B,n//B+1)])[:n]
    e=np.cumprod(1+r[idx]); bc.append(e[-1]**(252/n)-1); bm.append((e/np.maximum.accumulate(e)-1).min())
out['boot']=dict(cagr=np.percentile(bc,[5,25,50,75,95]).tolist(),mdd=np.percentile(bm,[5,25,50,75,95]).tolist())
json.dump(out,open('pro_robust.json','w'),ensure_ascii=False,default=float)
for k in ['sens','params','periods','drop']:
    print('==',k)
    for d in out[k]: print(f"{d['label']:<36} {d['CAGR']*100:6.1f}% {d['MDD']*100:6.1f}% Sh {d['Sharpe']:.2f} N {d['N']} 勝 {d['Win']*100:.0f}% PF {d['PF']:.2f}")
for d in cap: print(f"cap {d['cap0']/1e4:.0f}萬 mp {d['mp']:.0%} {d['CAGR']*100:.1f}% {d['MDD']*100:.1f}% Sh {d['Sharpe']:.2f} 期末 {d['final']/1e4:,.0f}萬")
print('random',np.percentile(rs[:,0],[5,50,95]).round(3),np.percentile(rs[:,1],[5,50,95]).round(3),np.percentile(rs[:,2],[5,50,95]).round(2),'pct',(rs[:,0]<out['sens'][0]['CAGR']).mean())
print('boot',np.round(out['boot']['cagr'],3),np.round(out['boot']['mdd'],3))
