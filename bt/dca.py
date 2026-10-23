from engine import *
import json
from scipy.optimize import brentq
P=prep(); NEW=dict(min_to=3e6,excl_ind=('生技醫療業','通信網路業'))
tx=P['taiex']
def xirr(dates,cfs):
    t=np.array([(d-dates[0]).days/365.25 for d in dates])
    f=lambda r: np.sum(cfs/(1+r)**t)
    return brentq(f,-0.99,10)
out=[]
for win,(a,b) in {'全期':('2016-01-01','2026-12-31'),'2016–2020':('2016-01-01','2020-12-31'),'2021–2026':('2021-01-01','2026-12-31')}.items():
  cap0=100_000
  for m in [0,5000,10000,20000]:
    for mf in [20]:
        E,T,N,_=run(P,dict(NEW,cap0=cap0,contrib=m,minfee=mf,start=a,end=b)); F=E.attrs['flows']
        inv=cap0+F.sum()
        # 時間加權報酬
        r=(E-F)/E.shift(1)-1; r=r.dropna(); twr=(1+r).prod()**(365.25/(E.index[-1]-E.index[0]).days)-1
        dd=(1+r).cumprod(); mdd=(dd/dd.cummax()-1).min()
        # 金額回撤
        ddamt=(E-E.cummax()).min()
        cf_dates=[E.index[0]]+list(F[F>0].index)+[E.index[-1]]
        cfs=np.array([-cap0]+list(-F[F>0].values)+[E.iloc[-1]])
        irr=xirr(cf_dates,cfs)
        # 同樣金流投入大盤(價格指數)
        u=cap0/tx.loc[E.index[0]]; 
        for d,v in F[F>0].items(): u+=v/tx.loc[d]
        bm=u*tx.loc[E.index[-1]]
        ye=E.resample('YE').last(); inv_y=(cap0+F.cumsum()).resample('YE').last()
        # 大盤年底
        ub=(pd.Series(cap0/tx.loc[E.index[0]],index=[E.index[0]]))
        units=(F/tx.loc[E.index]).cumsum()+cap0/tx.loc[E.index[0]]; bmv=(units*tx.loc[E.index]).resample('YE').last()
        o=dict(win=win,cap0=cap0,m=m,minfee=mf,inv=inv,final=E.iloc[-1],profit=E.iloc[-1]-inv,irr=irr,twr=twr,mdd=mdd,ddamt=ddamt,
               ddamt_date=str((E-E.cummax()).idxmin().date()),bm=bm,N=len(T),
               years=[[int(d.year),float(v),float(inv_y.loc[d]),float(bmv.loc[d])] for d,v in ye.items()],
               curve=[[d.strftime('%Y-%m-%d'),float(v),float(iv)] for d,v,iv in zip(E.index[::5],E.values[::5],(cap0+F.cumsum()).values[::5])])
        out.append(o)
        print(win,f"本金{cap0/1e4:.0f}萬 月投{m:>5} 最低費{mf:>2} 投入 {inv/1e4:6.1f}萬 期末 {E.iloc[-1]/1e4:8.1f}萬 獲利 {(E.iloc[-1]-inv)/1e4:8.1f}萬 IRR {irr*100:5.1f}% TWR {twr*100:5.1f}% MDD {mdd*100:5.1f}% 最大金額回撤 {ddamt/1e4:6.1f}萬({o['ddamt_date']}) 大盤同金流 {bm/1e4:6.1f}萬 N {len(T)}")
json.dump(out,open('dca.json','w'),ensure_ascii=False,default=float)
