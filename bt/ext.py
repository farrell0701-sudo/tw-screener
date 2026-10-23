from engine import *
import json
P=prep(); NEW=dict(min_to=3e6,excl_ind=('生技醫療業','通信網路業'))
E,T,N,op=run(P,NEW); cal=E.index; tx=P['taiex'].loc[cal]
o={}
def yr(s):
    y=s.resample('YE').last(); y0=pd.concat([pd.Series([s.iloc[0]],index=[s.index[0]-pd.Timedelta(days=1)]),y]); return y0.pct_change().dropna()
ys,yb=yr(E),yr(tx); Y=pd.DataFrame({'s':ys,'b':yb}); Y.index=Y.index.year
F=Y.loc[2016:2025]
o['annual_corr']=F.s.corr(F.b); o['annual_corr_all']=Y.s.corr(Y.b)
b,a=np.polyfit(F.b,F.s,1); o['annual_beta']=b; o['annual_alpha']=a
o['excess_mean']=(F.s-F.b).mean(); o['excess_median']=(F.s-F.b).median()
o['ratio_mean']=(F.s/F.b)[F.b>0].mean()
o['mean_s']=F.s.mean(); o['mean_b']=F.b.mean()
o['beat_years']=int((F.s>F.b).sum())
o['excess_ex2023']=(F.s-F.b).drop(2023).mean()
# 日/月
rd=E.pct_change().dropna(); bd=tx.pct_change().dropna()
m=E.resample('ME').last().pct_change().dropna(); mb=tx.resample('ME').last().pct_change().dropna()
o['daily_corr']=rd.corr(bd); o['monthly_corr']=m.corr(mb)
bm,am=np.polyfit(mb,m,1); o['monthly_beta']=bm; o['alpha_ann']=(1+am)**12-1
bdd,add=np.polyfit(bd,rd,1); o['daily_beta']=bdd
te=(m-mb).std()*np.sqrt(12); o['te']=te; o['ir']=(m-mb).mean()*12/te
up=mb>0; dn=mb<0
o['up_capture']=((1+m[up]).prod()**(12/up.sum())-1)/((1+mb[up]).prod()**(12/up.sum())-1)
o['down_capture']=((1+m[dn]).prod()**(12/dn.sum())-1)/((1+mb[dn]).prod()**(12/dn.sum())-1)
o['up_months']=int(up.sum()); o['down_months']=int(dn.sum())
o['avg_m_up']=m[up].mean(); o['avg_b_up']=mb[up].mean(); o['avg_m_dn']=m[dn].mean(); o['avg_b_dn']=mb[dn].mean()
o['win_dn_months']=float((m[dn]>0).mean()); o['beat_m']=float((m>mb).mean())
o['pos_months']=float((m>0).mean()); o['best_m']=float(m.max()); o['worst_m']=float(m.min())
o['best_m_d']=m.idxmax().strftime('%Y-%m'); o['worst_m_d']=m.idxmin().strftime('%Y-%m')
o['skew']=float(rd.skew()); o['kurt']=float(rd.kurt())
o['var95']=float(rd.quantile(.05)); o['cvar95']=float(rd[rd<=rd.quantile(.05)].mean())
o['b_var95']=float(bd.quantile(.05))
dd=E/E.cummax()-1; o['ulcer']=float(np.sqrt((dd**2).mean()))
ddb=tx/tx.cummax()-1; o['ulcer_b']=float(np.sqrt((ddb**2).mean()))
# 滾動 12 月
r12=E.resample('ME').last().pct_change(12).dropna(); b12=tx.resample('ME').last().pct_change(12).dropna()
o['r12_min']=float(r12.min()); o['r12_max']=float(r12.max()); o['r12_pos']=float((r12>0).mean()); o['r12_beat']=float((r12>b12).mean()); o['r12_med']=float(r12.median())
o['r12']=[[d.strftime('%Y-%m'),float(v),float(b12.loc[d])] for d,v in r12.items()]
# 滾動 36 月年化
r36=(E.resample('ME').last().pct_change(36)+1)**(1/3)-1; r36=r36.dropna(); o['r36_min']=float(r36.min()); o['r36_med']=float(r36.median())
# 滾動相關(12M)
rc=m.rolling(12).corr(mb).dropna(); o['rollcorr']=[[d.strftime('%Y-%m'),float(v)] for d,v in rc.items()]
# 曝險：大盤下跌月平均持股
Nm=(N/8).resample('ME').mean().loc[m.index]
o['expo_up']=float(Nm[up].mean()); o['expo_dn']=float(Nm[dn].mean())
# 大盤上漲/下跌年份分組
o['years']=[[int(i),float(r.s),float(r.b)] for i,r in Y.iterrows()]
# 年度最大回撤 大盤
o['bench']=stats(tx)
# 市值/產業分布
names=P['lst'].set_index('code'); ind=names.industry
sh=P['shares']; T['cap']=T.code.map(lambda c: sh.get(c,np.nan)*10/1e8)
T['ind']=T.code.map(ind); T['pnl']=T.proceeds-T.cost
g=T.groupby('ind').agg(n=('ret','size'),pnl=('pnl','sum')).sort_values('pnl',ascending=False); g['share']=g.pnl/T.pnl.sum()
o['ind']=[[k,int(r.n),float(r.share)] for k,r in g.head(8).iterrows()]
o['n_ind']=int(T.ind.nunique()); o['n_stocks']=int(T.code.nunique())
# 市值分組(用 現價*股數 近似)
o['cap_buckets']=T.assign(b=pd.cut(T.cap,[0,30,100,1e9],labels=['股本<30億','30–100億','≥100億'])).groupby('b',observed=True).agg(n=('ret','size'),avg=('ret','mean'),pnl=('pnl','sum')).assign(share=lambda d:d.pnl/T.pnl.sum()).reset_index().astype({'b':str}).to_dict('records')
# 持有天數分組
T['hb']=pd.cut(T.days,[0,10,30,60,120,1000],labels=['≤10天','11–30天','31–60天','61–120天','>120天'])
o['hold']=T.groupby('hb',observed=True).agg(n=('ret','size'),win=('ret',lambda x:(x>0).mean()),avg=('ret','mean'),pnl=('pnl','sum')).assign(share=lambda d:d.pnl/T.pnl.sum()).reset_index().astype({'hb':str}).to_dict('records')
# 每年交易
T['y']=pd.to_datetime(T.exit).dt.year
o['tyear']=T.groupby('y').agg(n=('ret','size'),win=('ret',lambda x:(x>0).mean()),avg=('ret','mean')).reset_index().to_dict('records')
# 週轉
o['turnover']=float(T.cost.sum()/E.mean()/((cal[-1]-cal[0]).days/365.25))
o['trades_per_year']=len(T)/((cal[-1]-cal[0]).days/365.25)
# 成本拖累
o['cost_drag']=None
# 股息貢獻估計: 還原 vs 原始
json.dump(o,open('ext.json','w'),ensure_ascii=False,default=float)
for k,v in o.items():
    if k not in ('r12','rollcorr','years','ind','cap_buckets','hold','tyear','bench'): print(k, round(v,4) if isinstance(v,float) else v)
print(o['years']); print(o['ind']); print(o['cap_buckets']); print(o['hold']); print(o['tyear'])
