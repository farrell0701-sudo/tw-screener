"""逐日 overlap 特徵（全市場），全部只用當天以前的資料。
輸出 /home/claude/bt/struct/daily.pkl：dict(feature -> DataFrame[日期 × 代號])"""
import os, sys, time
os.environ.setdefault('RAWSRC', 'finmind'); os.environ.setdefault('DELIST', '1')
sys.path.insert(0, '/home/claude/bt'); sys.path.insert(0, '/home/claude/bt/struct')
import numpy as np, pandas as pd
from load import load_all
from scan import overlap_lines

OVH_A, OVH_B = 21, 250        # 上方壓力的回看範圍（根）
BAND = 1.10                   # 近 5 日高點往上 10% 的價格帶


def fix_jumps(a, th=0.11):
    """與 fac/panel.py 相同：上市 5 天後 >11% 的單日變動視為減資／資料錯誤，報酬設 0 並向前等比例調整"""
    r = a.C / a.C.shift(1) - 1
    bad = (r.abs() > th) & (np.arange(len(a)) > 5)
    adj = (1 + r.where(bad, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    a = a.copy()
    for k in ('O', 'H', 'L', 'C'): a[k] = a[k] * adj
    return a


def stock_daily(a):
    a = fix_jumps(a)
    H, L, C, V = (a[k].values.astype(float) for k in ('H', 'L', 'C', 'V'))
    n = len(C)
    near, far = overlap_lines(H, L)
    s = pd.DataFrame({'near': near, 'far': far, 'H': H, 'L': L, 'C': C, 'V': V})
    near5 = s.near.rolling(5).mean(); far5 = s.far.rolling(5).mean()
    old_top = s.H.shift(21).rolling(100).quantile(0.9)
    base_before = far5.shift(15).rolling(11).max()
    lo5 = s.L.rolling(5).min()
    trans = (near5 >= 0.35) & (far5 <= 0.10) & (near5 > near5.shift(3)) & (lo5 >= old_top * 0.98) & (base_before >= 0.30)
    # 上方壓力：前 21～250 根中，K 棒區間和 [近5日高, ×1.10] 有交集者的成交量占比
    hi5 = s.H.rolling(5).max().values
    vol_hit = np.zeros(n); vol_tot = np.zeros(n)
    for k in range(OVH_A, OVH_B + 1):
        hk = np.r_[np.full(k, np.nan), H[:-k]]; lk = np.r_[np.full(k, np.nan), L[:-k]]; vk = np.r_[np.zeros(k), V[:-k]]
        m = (np.minimum(hk, hi5 * BAND) - np.maximum(lk, hi5)) > 0
        vol_hit += np.where(m, vk, 0); vol_tot += vk
    ovh = np.where(np.arange(n) >= OVH_B, vol_hit / np.where(vol_tot > 0, vol_tot, np.nan), np.nan)
    r20 = s.C / s.C.shift(20) - 1
    drop20 = r20.rolling(300, min_periods=150).min()          # 近 300 根內最急的 20 日跌幅
    lo_ratio = s.L.shift(20).rolling(230).min() / s.L.shift(250).rolling(250).min()
    return dict(near5=near5.values, far5=far5.values, trans=trans.values.astype(float),
                trans20=trans.rolling(20, min_periods=1).max().values, ovh=ovh,
                drop20=drop20.values, lo_ratio=lo_ratio.values)


if __name__ == '__main__':
    t0 = time.time()
    lst, tx, cal, S, _ = load_all()
    out = {}
    for i, (code, a) in enumerate(S.items(), 1):
        if len(a) < 300: continue
        d = stock_daily(a)
        for k, v in d.items():
            out.setdefault(k, {})[code] = pd.Series(v, index=a.index)
        if i % 500 == 0: print(f'{i}/{len(S)}  {time.time() - t0:.0f} 秒', flush=True)
    W = {k: pd.DataFrame(v).reindex(cal).astype('float32') for k, v in out.items()}
    pd.to_pickle(W, '/home/claude/bt/struct/daily.pkl')
    print('完成', {k: v.shape for k, v in W.items()}, f'{time.time() - t0:.0f} 秒')
