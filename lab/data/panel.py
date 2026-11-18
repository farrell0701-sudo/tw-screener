# 建面板：tw-data（固定標籤）→ lab/cache/panel/*.npy（日期 × 證券）
# 用法：python lab/data/panel.py            建面板＋寫品質報告 lab/data/PANEL_REPORT.md
#       from lab.data.panel import load      讀取（記憶體映射，不整份載入）
# 報告只含彙總數字，不含任何原始資料（tw-screener 是公開 repo）。
import os, re, sys, json, time, subprocess, tomllib
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / 'lab'
OUT = LAB / 'cache' / 'panel'
CFG = tomllib.loads((LAB / 'config.toml').read_text(encoding='utf-8'))

# 欄位：名稱 → (dtype, 預設值)
F32 = ['O', 'H', 'L', 'C', 'V', 'AMT', 'LIMU', 'LIMD']        # 未還原價、量（股）、成交值（元）、漲跌停價
F64 = ['AF', 'AF_C']                                          # 還原因子：完整版、相容版（偏誤 ②）
BOOL = ['EX', 'TRD', 'SUSP', 'EMG', 'NOLIM', 'LUL', 'LDL', 'OSUS', 'DEF', 'DISP', 'MISS']
# EX 有資料列｜TRD 有成交且未停牌且收盤>0｜SUSP 停牌｜EMG 興櫃｜NOLIM 不設漲跌幅｜LUL/LDL 漲停/跌停鎖死
# OSUS 開盤價可疑｜DEF 瑕疵日｜DISP 處置｜MISS price_status＝missing


def twdata_path():
    p = Path(os.environ.get('TWDATA') or (ROOT / CFG['data']['twdata']))
    return p.resolve()


def check_tag(p):
    tag = subprocess.run(['git', '-C', str(p), 'describe', '--tags', '--exact-match'],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(p), 'status', '--porcelain', '--', 'derived'],
                           capture_output=True, text=True).stdout.strip()
    if tag != CFG['data']['tag'] or dirty:
        sys.exit(f'tw-data 不是乾淨的 {CFG["data"]["tag"]} 標籤（目前：{tag or "無標籤"}，derived 有變更：{bool(dirty)}）')
    return tag


def peak_mem_mb():
    try:
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                        'PagefileUsage', 'PeakPagefileUsage')]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb): return float('nan')
        return c.PeakWorkingSetSize / 2**20
    except Exception:
        try:
            import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return float('nan')


def pk_industry():
    """相容模式產業名單：fl_small.py 寫死的 PK 字串（產業:代碼,代碼|...）"""
    src = (LAB / 'recovered' / 'fl_small.py').read_text(encoding='utf-8')
    s = re.search(r'^PK = "(.*)"$', src, re.M).group(1)
    return {c: k for blk in s.split('|') for k, cs in [blk.split(':')] for c in cs.split(',')}


def build():
    t0 = time.time(); R = twdata_path(); tag = check_tag(R); D = R / 'derived'
    start = pd.Timestamp(CFG['data']['start'])
    M = pd.read_parquet(D / 'security_master.parquet')
    M = M[M['security_class'].isin(CFG['data']['classes'])].copy()
    M = M[pd.to_datetime(M['last_price_date']) >= start].sort_values('security_key').reset_index(drop=True)
    keys = M['security_key'].tolist(); kix = {k: i for i, k in enumerate(keys)}

    IX = pd.read_parquet(D / 'index.parquet'); IX['date'] = pd.to_datetime(IX['date'])
    tx = IX[IX['index_id'] == 'TAIEX'].set_index('date').sort_index()
    years = [y for y in range(start.year, 2100) if (D / 'price_adj' / f'{y}.parquet').exists()]
    # 交易日：TAIEX 有值的日子 ∪ 股票有價的日子（之後檢查兩者一致）
    pdays = set()
    for y in years:
        d = pq.read_table(D / 'price_adj' / f'{y}.parquet', columns=['date']).column('date').to_pandas()
        pdays |= set(pd.to_datetime(d.unique()))
    tdays = set(tx.index[tx.index >= start])
    dates = pd.DatetimeIndex(sorted(pdays | tdays)); dates = dates[dates >= start]
    T, N = len(dates), len(keys)

    OUT.mkdir(parents=True, exist_ok=True)
    A = {}
    for f in F32: A[f] = np.lib.format.open_memmap(OUT / f'{f}.npy', 'w+', np.float32, (T, N)); A[f][:] = np.nan
    for f in BOOL: A[f] = np.lib.format.open_memmap(OUT / f'{f}.npy', 'w+', np.bool_, (T, N))
    EF = np.ones((T, N)); AFS = np.full((T, N), np.nan)     # 當日事件因子、tw-data 給的 af

    pcols = ['security_key', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'price_status', 'event_factor', 'af']
    tcols = ['security_key', 'date', 'has_trade', 'suspended', 'disposition', 'emerging', 'no_price_limit',
             'limit_up', 'limit_down', 'limit_up_locked', 'limit_down_locked', 'open_suspect', 'defect_day']
    nrows = 0
    for y in years:
        p = pd.read_parquet(D / 'price_adj' / f'{y}.parquet', columns=pcols)
        p = p[p['security_key'].isin(kix)]
        tr = pd.read_parquet(D / 'tradable' / f'{y}.parquet', columns=tcols)
        p = p.merge(tr, on=['security_key', 'date'], how='left'); del tr
        p['date'] = pd.to_datetime(p['date']); p = p[p['date'] >= start]
        ti = dates.get_indexer(p['date']); ni = p['security_key'].map(kix).to_numpy()
        assert (ti >= 0).all()
        nrows += len(p)
        for f, c in [('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('V', 'volume'), ('AMT', 'value'),
                     ('LIMU', 'limit_up'), ('LIMD', 'limit_down')]:
            A[f][ti, ni] = p[c].to_numpy(np.float32, na_value=np.nan)
        b = lambda c: p[c].fillna(False).to_numpy(bool)
        A['EX'][ti, ni] = True
        A['SUSP'][ti, ni] = b('suspended')
        A['TRD'][ti, ni] = b('has_trade') & ~b('suspended') & (p['close'].fillna(0).to_numpy() > 0)
        for f, c in [('EMG', 'emerging'), ('NOLIM', 'no_price_limit'), ('LUL', 'limit_up_locked'), ('LDL', 'limit_down_locked'),
                     ('OSUS', 'open_suspect'), ('DEF', 'defect_day'), ('DISP', 'disposition')]:
            A[f][ti, ni] = b(c)
        A['MISS'][ti, ni] = (p['price_status'] == 'missing').to_numpy()
        EF[ti, ni] = p['event_factor'].fillna(1.0).to_numpy(); AFS[ti, ni] = p['af'].to_numpy(np.float64, na_value=np.nan)
        del p
    for f in list(A): A[f].flush()

    # 還原因子：af_t ＝ Π_{s>t} 事件因子_s（最新一日＝1）。由事件因子重建，並與 tw-data 給的 af 比對。
    def af_from(ef):
        c = np.cumprod(ef[::-1], axis=0)[::-1]        # Π_{s≥t}
        return np.vstack([c[1:], np.ones((1, ef.shape[1]))])   # Π_{s>t}
    AF = af_from(EF)
    # 面板之後的事件（無）＋面板之前（不影響 t≥start）→ 應與 tw-data af 一致
    m = ~np.isnan(AFS)
    af_err = float(np.nanmax(np.abs(AF[m] / AFS[m] - 1))) if m.any() else float('nan')

    # 相容版（偏誤 ②）：2016/7 以前的股利事件不還原
    E = pd.read_parquet(D / 'events.parquet')
    E = E[E['security_key'].isin(kix) & (E['status'] == 'ok')].copy(); E['apply_date'] = pd.to_datetime(E['apply_date'])
    E = E[E['apply_date'] >= start]
    # 檢查：同一證券同一日事件表因子乘積 ＝ 價格表 event_factor
    g = E.groupby(['security_key', 'apply_date'])['factor'].prod().reset_index()
    gi = dates.get_indexer(g['apply_date']); gn = g['security_key'].map(kix).to_numpy(); ok = gi >= 0
    ev_mis = int((np.abs(EF[gi[ok], gn[ok]] / g['factor'].to_numpy()[ok] - 1) > 1e-6).sum()); ev_off = int((~ok).sum())
    ef_rows = int((np.abs(EF - 1) > 1e-12).sum())
    cc = CFG['compat']
    X = E[E['source'].isin(cc['div_sources']) & (E['apply_date'] < pd.Timestamp(cc['no_div_before']))]
    xg = X.groupby(['security_key', 'apply_date'])['factor'].prod().reset_index()
    xi = dates.get_indexer(xg['apply_date']); xn = xg['security_key'].map(kix).to_numpy(); k_ = xi >= 0
    EFC = EF.copy(); EFC[xi[k_], xn[k_]] /= xg['factor'].to_numpy()[k_]
    AF_C = af_from(EFC); del EFC
    np.save(OUT / 'AF.npy', AF); np.save(OUT / 'AF_C.npy', AF_C)

    # 中繼資料
    pk = pk_industry()
    M['ind_pk'] = M['stock_id'].map(pk)
    dl = pd.to_datetime(M['delist_date']); lt = pd.to_datetime(M['last_trade_date'])
    # 舊引擎缺「2016 後不再存在」的股票：已下市＋inactive_unexplained 都算
    lp = pd.to_datetime(M['last_price_date'])
    M['bias1_exclude'] = (M['status'] != 'active') & (dl.fillna(lp) >= pd.Timestamp(cc['delist_from']))
    keep = ['security_key', 'stock_id', 'segment', 'name', 'market_current', 'industry', 'ind_pk', 'status',
            'first_listed_date', 'last_listed_date', 'first_trade_date', 'last_trade_date', 'last_price_date',
            'emerging_first_date', 'emerging_last_date', 'delist_date', 'bias1_exclude']
    M[keep].to_parquet(OUT / 'meta.parquet', index=False)
    np.save(OUT / 'dates.npy', dates.values.astype('datetime64[D]'))
    ixd = pd.DataFrame(index=dates)
    for iid in ('TAIEX', 'TPEx'):
        s = IX[IX['index_id'] == iid].set_index('date').sort_index()
        ixd[f'{iid}_P'] = s['price_index'].reindex(dates); ixd[f'{iid}_TR'] = s['total_return_index'].reindex(dates)
    ixd.reset_index(names='date').to_parquet(OUT / 'index.parquet', index=False)

    info = dict(tag=tag, built=time.strftime('%Y-%m-%d %H:%M'), T=T, N=N, start=str(dates[0].date()), end=str(dates[-1].date()),
                rows=nrows, af_max_rel_err=af_err, ev_mismatch=ev_mis, ev_off_calendar=ev_off, ef_rows=ef_rows,
                ev_ok=len(g), compat_div_events=int(k_.sum()),
                taiex_days_missing_in_panel=len(tdays - set(dates)), panel_days_without_taiex=len(set(dates) - tdays),
                pk_n=len(pk), pk_matched=int(M['ind_pk'].notna().sum()), seconds=round(time.time() - t0), peak_mb=round(peak_mem_mb()))
    (OUT / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


def load(fields=None, mmap=True):
    """回傳 dict：dates、meta、index 與指定欄位（預設全部），陣列為記憶體映射（唯讀）。"""
    fields = fields or (F32 + F64 + BOOL)
    out = {'dates': pd.DatetimeIndex(np.load(OUT / 'dates.npy')), 'meta': pd.read_parquet(OUT / 'meta.parquet'),
           'index': pd.read_parquet(OUT / 'index.parquet').set_index('date'),
           'info': json.loads((OUT / 'info.json').read_text(encoding='utf-8'))}
    for f in fields: out[f] = np.load(OUT / f'{f}.npy', mmap_mode='r' if mmap else None)
    return out


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    if '--report-only' not in sys.argv:
        print(json.dumps(build(), ensure_ascii=False, indent=1), flush=True)
    from panel_report import report
    report()
