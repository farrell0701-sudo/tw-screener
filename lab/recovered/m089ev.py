import pandas as pd, numpy as np
for w in ['cpold','cpnew']:
    M=pd.read_pickle(f'/home/claude/old/M089_{w}.pkl'); d=pd.DatetimeIndex(M['dates']); inp=M['inpos']
    segs=[('05-15','2005','2015')] if w=='cpold' else [('16-20','2016','2020'),('21-26','2021','2026')]
    for nm,lo,hi in segs:
        m=(d>=pd.Timestamp(lo))&(d<=pd.Timestamp(hi+'-12-31'))
        print(f'== {nm} 空頭持有日 {inp[m].sum()} 天（{inp[m].mean()*100:.0f}%）：只計空頭期間的累積報酬（扣成本）｜ 空頭期間年化 ｜ 空頭期間最大回撤')
        for k,r in M['R'].items():
            x=pd.Series(r[m&inp]); e=(1+x).cumprod()
            if len(x)==0: continue
            print(f'   {k:10s} {e.iloc[-1]-1:+.1%} ｜ {e.iloc[-1]**(252/len(x))-1:+.1%} ｜ {(e/e.cummax()-1).min():.1%}')
        # 逐段空頭
        runs=[]; i=0; idx=np.where(m)[0]
        while i<len(idx):
            t=idx[i]
            if inp[t]:
                j=i
                while j+1<len(idx) and inp[idx[j+1]]: j+=1
                if j-i+1>=20: runs.append((idx[i],idx[j]))
                i=j+1
            else: i+=1
        print('   逐段（≥20 日）：期間 ｜ A 最陡全部 前半/後半/全段 ｜ B 前6 全段 ｜ C 強勢池 全段 ｜ D 指數 全段')
        for a,b in runs:
            mid=(a+b)//2; f=lambda k,s,e:(1+pd.Series(M['R'][k][s:e+1])).prod()-1
            print(f'   {d[a]:%Y-%m-%d}~{d[b]:%Y-%m-%d}（{b-a+1}日）｜ {f("A 最陡全部",a,mid):+.1%}/{f("A 最陡全部",mid+1,b):+.1%}/{f("A 最陡全部",a,b):+.1%} ｜ {f("B 最陡T_ld前6",a,b):+.1%} ｜ {f("C 強勢池全部",a,b):+.1%} ｜ {f("D 加權指數",a,b):+.1%}')
