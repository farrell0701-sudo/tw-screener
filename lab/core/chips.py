# 整理層籌碼 → 面板格子（日期 × 證券）→ lab/cache/chips/*.npy
# 用法：from chips import crowd ；crowd(P, F, W) → (TURN_W, MUSE)，皆為 T×N float32。L-114。
# 規則（tw-data CONTRACT 第 6 節）：籌碼以資料日 date 收盤後可用 → 訊號日 t 用 date ≤ t。
# 同代碼多段（代碼重用）時，資料放到當日存在（EX）的那一段。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data'))
from panel import twdata_path

OUT = LAB / 'cache' / 'chips'


def _staged(name, cols):
    d = twdata_path() / 'staged' / name
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(d.glob('*.parquet'))], ignore_index=True)
    df['date'] = pd.to_datetime(df['date']); return df


def to_grid(df, val, P):
    dates = P['dates']; M = P['meta']; T, N = len(dates), len(M)
    cols_of = {}
    for j, s in enumerate(M['stock_id'].astype(str)): cols_of.setdefault(s, []).append(j)
    A = np.full((T, N), np.nan, np.float32)
    ti = dates.get_indexer(df['date']); keep = ti >= 0
    df = df[keep]; ti = ti[keep]
    sid = df['stock_id'].astype(str).to_numpy(); v = df[val].to_numpy(np.float64)
    EX = np.asarray(P['EX'])
    for s, idx in pd.Series(np.arange(len(df))).groupby(sid):
        cs = cols_of.get(s)
        if not cs: continue
        rows = idx.to_numpy()
        if len(cs) == 1: A[ti[rows], cs[0]] = v[rows]
        else:
            for c in cs:
                m = EX[ti[rows], c]; A[ti[rows][m], c] = v[rows][m]
    return A


def build(P):
    """融資使用率 MUSE、發行股數 SHR（向前填補最多 5 日）→ 快取。"""
    OUT.mkdir(parents=True, exist_ok=True)
    mg = _staged('TaiwanStockMarginPurchaseShortSale', ['date', 'stock_id', 'MarginPurchaseTodayBalance', 'MarginPurchaseLimit'])
    bal = pd.DataFrame(to_grid(mg, 'MarginPurchaseTodayBalance', P)).ffill(limit=5).to_numpy()
    lim = pd.DataFrame(to_grid(mg, 'MarginPurchaseLimit', P)).ffill(limit=5).to_numpy(); del mg
    with np.errstate(all='ignore'):
        MUSE = np.where(lim > 0, bal / lim, np.nan).astype(np.float32)
    sh = _staged('TaiwanStockShareholding', ['date', 'stock_id', 'NumberOfSharesIssued'])
    SHR = pd.DataFrame(to_grid(sh, 'NumberOfSharesIssued', P)).ffill(limit=5).to_numpy().astype(np.float32)
    SHR[~(SHR > 0)] = np.nan
    np.save(OUT / 'MUSE.npy', MUSE); np.save(OUT / 'SHR.npy', SHR)
    (OUT / 'info.txt').write_text(f"panel {P['info'].get('built')} tag {P['info'].get('tag')} shape {MUSE.shape}\n", encoding='utf-8')


def crowd(P, F, W=20):
    """回傳 (TURN_W, MUSE)。TURN_W＝近 W 個市場交易日的平均成交股數（只計有效日，至少 0.75W 日）÷ 發行股數。"""
    need = [OUT / 'MUSE.npy', OUT / 'SHR.npy']
    ok = all(p.exists() for p in need)
    if ok:
        MUSE = np.load(need[0], mmap_mode='r')
        ok = MUSE.shape == P['C'].shape and (OUT / 'info.txt').read_text(encoding='utf-8').startswith(f"panel {P['info'].get('built')} ")
    if not ok: build(P)
    MUSE = np.load(OUT / 'MUSE.npy'); SHR = np.load(OUT / 'SHR.npy')
    V = np.where(np.asarray(F['VALID']), np.asarray(P['V'], np.float32), np.nan)
    avg = pd.DataFrame(V).rolling(W, min_periods=int(np.ceil(0.75 * W))).mean().to_numpy(np.float32)
    with np.errstate(all='ignore'):
        TURN = (avg / SHR).astype(np.float32)
    return TURN, MUSE
