# L-113B：把舊引擎當時的資料（tw-data 分支 old-data、history-data，已解壓到 lab/cache/old/d、h）組成與 panel.load() 相同格式的記憶體面板。
# 依 lab/recovered/AUDIT_BASE.md 第 2 節與 old_test.py。只在本機使用，不寫入 repo。
import glob, os
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
OLD = LAB / 'cache' / 'old'


def _read_dir(d, which):
    fr = {}
    for f in glob.glob(str(d / '[0-9]*.csv')):
        c = os.path.basename(f)[:-4]
        if len(c) != 4 or c.startswith('0'): continue
        x = pd.read_csv(f, parse_dates=['date' if which == 'old' else 'Date'])
        if len(x) < 30: continue
        if which == 'new':
            x = x.rename(columns={'Date': 'date', 'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'AdjClose': 'ac', 'Volume': 'v'})
        fr[c] = x.drop_duplicates('date', keep='last').set_index('date')
    return fr


def build(which='old', industry='list'):
    """which：old（2004–2016/06，old-data）或 new（2015 起，history-data）。industry：list（stock_list.csv）或 pk（fl_small.py 名單）。"""
    d = OLD / ('d' if which == 'old' else 'h')
    fr = _read_dir(d, which); cols = sorted(fr)
    W = lambda k: pd.DataFrame({c: fr[c][k] for c in cols}).sort_index().astype(float)
    O, H, L, C, V = W('o'), W('h'), W('l'), W('c'), W('v')
    for X in (O, H, L, C): X[X <= 0] = np.nan
    O = O.where(C.notna())
    dates = C.index
    if which == 'old':
        # old_test.py：超過當時漲跌幅的單日跳動視為除權減資，向前調整（現金股利不還原）
        r = C / C.ffill().shift(1) - 1
        lim = pd.Series(np.where(dates < pd.Timestamp('2015-06-01'), 0.075, 0.105), index=dates)
        bad = r.abs().gt(lim, axis=0) & C.notna() & (C.notna().cumsum() > 5)
        k = (1 + r.where(bad, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
        n_fix = int(bad.sum().sum())
    else:
        k = (W('ac') / C).reindex(index=dates); n_fix = 0
    ixf = pd.read_csv(d / 'TAIEX.csv')
    dc = 'date' if 'date' in ixf.columns else 'Date'; cc = 'c' if 'c' in ixf.columns else 'Close'
    ix = ixf.assign(**{dc: pd.to_datetime(ixf[dc])}).set_index(dc)[cc].astype(float)
    ix = ix[~ix.index.duplicated()].reindex(dates).ffill()
    lst = pd.read_csv(OLD / 'h' / 'stock_list.csv', dtype={'code': str}).set_index('code')
    from panel import pk_industry
    pk = pk_industry()
    ind_list = lst['industry'].reindex(cols).to_numpy() if 'industry' in lst.columns else np.full(len(cols), None)
    meta = pd.DataFrame({'security_key': cols, 'stock_id': cols, 'industry': ind_list, 'ind_pk': [pk.get(c) for c in cols],
                         'bias1_exclude': False, 'status': 'active', 'delist_date': pd.NaT})
    if industry == 'pk': meta['industry'] = meta['ind_pk']
    meta['ind_pk'] = meta['industry']           # 相容模式讀 ind_pk；這裡讓它等於所選產業來源
    f32 = lambda x: x.to_numpy(np.float32)
    TRD = (C.notna() & (V.fillna(0) > 0)).to_numpy()
    T, N = C.shape; z = np.zeros((T, N), bool)
    Kv = k.to_numpy(float)
    P = dict(C=f32(C), O=f32(O), H=f32(H), L=f32(L), V=f32(V), AMT=f32(C * V), AF=Kv, AF_C=Kv, TRD=TRD, EMG=z, LUL=z, LDL=z, OSUS=z,
             dates=pd.DatetimeIndex(dates), meta=meta, index=pd.DataFrame({'TAIEX_P': ix}), info=dict(built=f'old:{which}', tag='old-branches'))
    P['_n_fix'] = n_fix; P['_n_ind'] = int(pd.notna(meta['industry']).sum())
    return P
