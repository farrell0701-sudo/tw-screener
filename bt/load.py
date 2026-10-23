import pandas as pd, numpy as np, glob, os
H='/home/claude/hdata'
CUT='2026-10-16'
RAW=os.environ.get('RAWSRC','yahoo')   # 資料產生於 10/16 盤中 -> 這天不完整，丟掉
def load_all():
    lst=pd.read_csv(f'{H}/stock_list.csv',dtype={'code':str})
    t=pd.read_csv(f'{H}/TAIEX.csv',parse_dates=['Date']).set_index('Date')
    t=t[t.index<CUT]
    cal=t.index
    S={}
    stats=dict(files=0,rows=0,zero_vol=0,nan=0,spike=0)
    if os.environ.get('DELIST')=='1':
        dl=sorted(f[:-4] for f in os.listdir('/home/claude/dl_csv'))
        lst=pd.concat([lst,pd.DataFrame({'code':dl,'name':[c+'(下市)' for c in dl],'market':'DL','industry':None})],ignore_index=True)
    for code in lst.code:
        p=f'{H}/{code}.csv'
        if not os.path.exists(p): p=f'/home/claude/dl_csv/{code}.csv'
        if not os.path.exists(p): continue
        d=pd.read_csv(p,parse_dates=['Date']).set_index('Date')
        d=d[d.index<CUT]
        d=d[~d.index.duplicated()]
        stats['files']+=1; stats['rows']+=len(d)
        bad=d[['Open','High','Low','Close','AdjClose']].isna().any(axis=1)|(d.Close<=0)
        stats['nan']+=int(bad.sum()); d=d[~bad]
        z=d.Volume<=0; stats['zero_vol']+=int(z.sum()); d=d[~z]   # 零量=停牌/無交易
        if len(d)<60: continue
        f=d.AdjClose/d.Close
        a=pd.DataFrame({'O':d.Open*f,'H':d.High*f,'L':d.Low*f,'C':d.AdjClose,'V':d.Volume,'RC':d.Close})
        if RAW=='finmind':
            fp=f'/home/claude/rawx/{code}.csv'
            if os.path.exists(fp):
                r=pd.read_csv(fp,parse_dates=['date']).set_index('date')
                r=r[~r.index.duplicated()].reindex(a.index)
                stats['fm_hit']=stats.get('fm_hit',0)+int(r.c.notna().sum()); stats['fm_miss']=stats.get('fm_miss',0)+int(r.c.isna().sum())
                a['RC']=r.c.fillna(a.RC)              # 真正未還原收盤（缺值時退回 Yahoo）
                a['V']=r.v.astype(float).fillna(a.V)  # 真正成交股數
        a['H']=a[['H','O','C']].max(axis=1); a['L']=a[['L','O','C']].min(axis=1)
        S[code]=a
    return lst,t,cal,S,stats
