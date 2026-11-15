import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, mannwhitneyu
F={w:pd.read_pickle(f'/home/claude/old/M080F_{w}.pkl') for w in ['cpold','cpnew']}
S=['05-15','16-20','21-26']
FN=['T_ld百分位','ΔT_ld10百分位','20日漲幅','60日漲幅','ATR%','ER20','回撤比','距250日高/ATR','成交額(log)','量比5/20','重疊度短線','大盤20日','大盤>200MA']
for nm,lab in [('X','X1（6 檔等權）'),('V','V4.1（4 檔、剔除深回撤）')]:
    D=pd.concat([F[w][nm] for w in F]); D['seg']=np.where(D.date.dt.year<=2015,'05-15',np.where(D.date.dt.year<=2020,'16-20','21-26'))
    D['ex']=D.net-D['持有期大盤']; D['win']=D.net>0
    print(f'\n################ {lab}')
    print('== 一、贏輸統計（去重後每筆交易）')
    for s in S:
        X=D[D.seg==s]; w=X[X.win].net; l=X[~X.win].net; srt=np.sort(X.net.values)[::-1]; k=max(1,len(srt)//10)
        print(f' {s} n{len(X)} 勝率{X.win.mean()*100:.0f}% 平均贏{w.mean()*100:+.1f} 平均輸{l.mean()*100:+.1f} 盈虧比{w.mean()/-l.mean():.2f} 獲利因子{w.sum()/-l.sum():.2f} '
              f'平均{X.net.mean()*100:+.2f} 中位{X.net.median()*100:+.2f} ｜前10%交易占總損益{srt[:k].sum()/srt.sum()*100:.0f}% 拿掉前10%後平均{srt[k:].mean()*100:+.2f}'
              f' ｜持有 贏{X[X.win].hold.mean():.0f}/輸{X[~X.win].hold.mean():.0f}日 ｜輸家曾漲≥6% {(X[~X.win].mfe>=0.06).mean()*100:.0f}%')
        print(f'      報酬分位 P10 {np.percentile(X.net,10)*100:+.1f} P25 {np.percentile(X.net,25)*100:+.1f} P75 {np.percentile(X.net,75)*100:+.1f} P90 {np.percentile(X.net,90)*100:+.1f} P99 {np.percentile(X.net,99)*100:+.1f} ｜出場原因 '+', '.join(f'{k}{v*100:.0f}%' for k,v in X.rs.value_counts(normalize=True).items()))
    print('== 二、因子辨識力：AUC（贏vs輸，0.5=無）／IC（vs 淨報酬）／高三分位−低三分位 平均淨報酬差')
    res=[]
    for f in FN:
        line=f' {f:12s}'
        for s in S:
            X=D[(D.seg==s)].dropna(subset=[f])
            if X[f].nunique()<3: line+=' ‖ n/a'; continue
            try: auc=mannwhitneyu(X[X.win][f],X[~X.win][f]).statistic/(X.win.sum()*(~X.win).sum())
            except: auc=np.nan
            ic=spearmanr(X[f],X.net).correlation
            if f=='大盤>200MA': d=X[X[f]==1].net.mean()-X[X[f]==0].net.mean()
            else:
                q=pd.qcut(X[f].rank(method='first'),3,labels=False); d=X[q==2].net.mean()-X[q==0].net.mean()
            line+=f' ‖ {auc:.2f}/{ic:+.2f}/{d*100:+.1f}'; res.append((f,s,ic))
        print(line)
    print('== 三、診斷')
    for s in S:
        X=D[D.seg==s]
        r2m=spearmanr(X['持有期大盤'],X.net).correlation
        b=np.polyfit(X['持有期大盤'],X.net,1); R2=1-np.var(X.net-np.polyval(b,X['持有期大盤']))/np.var(X.net)
        mo=X.date.dt.to_period('M'); fe=X.net-X.groupby(mo).net.transform('mean'); R2m=1-np.var(fe)/np.var(X.net)
        aucM=mannwhitneyu(X[X.win]['持有期大盤'],X[~X.win]['持有期大盤']).statistic/(X.win.sum()*(~X.win).sum())
        print(f' {s} 持有期大盤報酬：AUC {aucM:.2f}、與淨報酬 Spearman {r2m:+.2f}、線性 R² {R2*100:.0f}%（β {b[0]:.2f}）｜進場月份固定效果 R² {R2m*100:.0f}%（每月平均 {X.groupby(mo).size().mean():.1f} 筆）')
    # 右尾 vs 中位：因子對「進前10%」與「是否賺錢」的區分
    print(' 右尾／中位分開看（全期，AUC：大贏家前10% vs 其餘 ｜ 賺 vs 賠）')
    D['top']=D.net>=D.groupby('seg').net.transform(lambda x: x.quantile(0.9))
    for f in ['T_ld百分位','ΔT_ld10百分位','20日漲幅','ATR%','回撤比','重疊度短線']:
        X=D.dropna(subset=[f])
        a1=mannwhitneyu(X[X.top][f],X[~X.top][f]).statistic/(X.top.sum()*(~X.top).sum()); a2=mannwhitneyu(X[X.win][f],X[~X.win][f]).statistic/(X.win.sum()*(~X.win).sum())
        print(f'   {f:12s} {a1:.2f} ｜ {a2:.2f}')
    # 範圍限縮
    iq=F['cpnew']['_uni_iqr']; print(' 範圍限縮（交易樣本 IQR ÷ 可交易池 IQR，越小代表選完後剩下的變異越少）')
    print('   '+'，'.join(f'{f} {(D[f].quantile(.75)-D[f].quantile(.25))/iq[f]:.2f}' for f in ['T_ld百分位','ΔT_ld10百分位','20日漲幅','60日漲幅','ATR%','ER20','距250日高/ATR','成交額(log)','量比5/20','重疊度短線'] if f in iq))
    # 雜訊上限
    ics=np.array([abs(r[2]) for r in res if r[2]==r[2]]); n=len(D)/3
    print(f'   每段約 {n:.0f} 筆；|IC| 中位 {np.median(ics):.3f}；IC 要達 t=2 需 |IC| ≥ {2/np.sqrt(n):.3f}；以 |IC| 中位，需約 {4/np.median(ics)**2:.0f} 筆')
