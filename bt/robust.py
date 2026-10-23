from engine import *
import json, numpy as np
P=prep()
def S(cfg,label):
    E,T,N,_=run(P,cfg); s=stats(E,T)
    return dict(label=label,CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],N=s.get('N',0),Win=s.get('Win',0),PF=s.get('PF',0),Final=s['Final'])
out={}
base=S({},'基準')
sens=[base]
for cfg,l in [({'slip':0.002},'滑價 0.2%/邊'),({'slip':0.003},'滑價 0.3%/邊'),({'limitdown_block':True},'跌停鎖死賣不掉→隔日再賣'),
 ({'min_to':10e6},'流動性 20日均額≥1千萬'),({'min_to':30e6},'流動性 ≥3千萬'),({'min_to':50e6},'流動性 ≥5千萬'),
 ({'minscore':2},'評分 ≥2'),({'minscore':4},'評分 ≥4'),({'minscore':5},'評分 ≥5'),
 ({'atr_mult':3},'停損 ATR×3'),({'atr_mult':5},'停損 ATR×5'),({'be_trig':1.04},'保本觸發 4%'),({'be_trig':1.08},'保本觸發 8%'),
 ({'pos12':-0.10},'位置 −10%'),({'pos12':-0.15},'位置 −15%'),({'use_rescue':False},'不用救回組'),({'use_mkt':False},'不用大盤濾網'),
 ({'slots':5,'w':0.2},'5 格 × 20%'),({'slots':10,'w':0.1},'10 格 × 10%'),({'slots':12,'w':1/12},'12 格')]:
    sens.append(S(cfg,l))
out['sens']=sens
# 期間
per=[]
for a,b,l in [('2016-01-01','2020-12-31','2016–2020'),('2021-01-01','2026-12-31','2021–2026'),('2016-01-01','2022-12-31','2016–2022(不含AI行情)')]:
    per.append(S({'start':a,'end':b},l))
for y in range(2017,2023): per.append(S({'start':f'{y}-01-01'},f'{y} 起'))
out['periods']=per
# 去掉最大贏家
T0=pd.read_csv('trades_base.csv',dtype={'code':str}); T0['pnl']=T0.proceeds-T0.cost
top=T0.groupby('code').pnl.sum().sort_values(ascending=False).index.tolist()
drop=[]
for k in [1,3,5,10]:
    P2=dict(P); P2['sig']=P['sig'][~P['sig'].code.isin(top[:k])]
    E,T,N,_=run(P2); s=stats(E,T); drop.append(dict(label=f'移除獲利前 {k} 檔股票 ({",".join(top[:k][:5])}{"…" if k>5 else ""})',CAGR=s['CAGR'],MDD=s['MDD'],Sharpe=s['Sharpe'],N=s['N'],Win=s['Win'],PF=s['PF'],Final=s['Final']))
out['drop']=drop
# 隨機排序
rs=[]
for sd in range(300):
    E,T,N,_=run(P,{'rank':'random','seed':sd}); s=stats(E,T); rs.append([s['CAGR'],s['MDD'],s['Sharpe']])
rs=np.array(rs); out['random']=dict(cagr=rs[:,0].tolist(),mdd=rs[:,1].tolist(),sharpe=rs[:,2].tolist())
# 區塊 bootstrap
E=pd.read_csv('eq_base.csv',index_col=0,parse_dates=True).iloc[:,0]
r=E.pct_change().dropna().to_numpy(); n=len(r); rng=np.random.default_rng(1); B=20
bc=[];bm=[]
for _ in range(3000):
    idx=np.concatenate([np.arange(s,s+B) for s in rng.integers(0,n-B,n//B+1)])[:n]
    x=r[idx]; e=np.cumprod(1+x); bc.append(e[-1]**(252/n)-1); bm.append((e/np.maximum.accumulate(e)-1).min())
out['boot']=dict(cagr=np.percentile(bc,[5,25,50,75,95]).tolist(),mdd=np.percentile(bm,[5,25,50,75,95]).tolist())
json.dump(out,open('robust.json','w'),ensure_ascii=False,default=float,indent=1)
for k in ['sens','periods','drop']:
    for d in out[k]: print(f"{d['label']:<34} CAGR {d['CAGR']*100:6.1f}%  MDD {d['MDD']*100:6.1f}%  Sh {d['Sharpe']:.2f}  N {d['N']}  PF {d['PF']:.2f}")
print('random CAGR pct',np.percentile(rs[:,0],[5,50,95]).round(3),'MDD',np.percentile(rs[:,1],[5,50,95]).round(3),'Sharpe',np.percentile(rs[:,2],[5,50,95]).round(2))
print('boot',out['boot'])
