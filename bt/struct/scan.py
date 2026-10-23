"""全市場結構轉變（C→B→A）掃描：K 棒重疊辨識 + 特徵 + 成敗標籤
資料：FinMind 真實未還原價 + Yahoo 還原比例 + 2016 後下市股（環境變數 RAWSRC=finmind DELIST=1）
所有特徵只用事件當天（含）以前的資料；標籤用之後的資料。"""
import os, sys
os.environ.setdefault('RAWSRC', 'finmind'); os.environ.setdefault('DELIST', '1')
sys.path.insert(0, '/home/claude/bt')
import numpy as np, pandas as pd
from load import load_all

NEAR, FAR = 20, 120
COOLDOWN = 20


def overlap_lines(H, L):
    n = len(H); near = np.zeros(n); far = np.zeros(n)
    for k in range(1, FAR + 1):
        o = np.zeros(n, bool)
        o[k:] = (np.minimum(H[k:], H[:-k]) - np.maximum(L[k:], L[:-k])) > 0
        if k <= NEAR: near += o
        else: far += o
    near /= NEAR; far /= (FAR - NEAR)
    near[:FAR] = np.nan; far[:FAR] = np.nan
    return near, far


def band_density(H, L, V, t, lo, hi, a, b):
    """t 往前第 a～b 根中，K 棒區間和 [lo, hi] 有交集的比例，以及這些 K 棒的成交量占比"""
    s, e = max(0, t - b), t - a + 1
    if e <= s: return np.nan, np.nan
    h, l, v = H[s:e], L[s:e], V[s:e]
    m = (np.minimum(h, hi) - np.maximum(l, lo)) > 0
    return m.mean(), (v[m].sum() / v.sum() if v.sum() > 0 else np.nan)


def scan_stock(code, a, mk, tx):
    H, L, C, V, RC = (a[k].values.astype(float) for k in ('H', 'L', 'C', 'V', 'RC'))
    idx = a.index; n = len(C)
    if n < 300: return []
    near, far = overlap_lines(H, L)
    s = pd.DataFrame({'near': near, 'far': far, 'H': H, 'L': L}, index=idx)
    near5 = s.near.rolling(5).mean().values; far5 = s.far.rolling(5).mean().values
    old_top = s.H.shift(21).rolling(100).quantile(0.9).values
    base_before = pd.Series(far5).shift(15).rolling(11).max().values
    lo5 = s.L.rolling(5).min().values
    near5_up = near5 > pd.Series(near5).shift(3).values
    trans = (near5 >= 0.35) & (far5 <= 0.10) & near5_up & (lo5 >= old_top * 0.98) & (base_before >= 0.30)
    trans = np.nan_to_num(trans).astype(bool)
    Cs = pd.Series(C, index=idx)
    e = lambda sp: Cs.ewm(span=sp, adjust=False).mean().values
    e12, e20, e144, e169, e200 = e(12), e(20), e(144), e(169), e(200)
    slope = np.full(n, np.nan); slope[10:] = (e200[10:] / e200[:-10] - 1) * 10
    align = (e12 > e20) & (e20 > e144) & (e144 > e169) & (e169 > e200)
    run = np.zeros(n, int)
    for i in range(1, n): run[i] = run[i - 1] + 1 if align[i] else 0
    pc = np.r_[np.nan, C[:-1]]
    tr = np.nanmax(np.c_[H - L, np.abs(H - pc), np.abs(L - pc)], axis=1)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    amt20 = pd.Series(RC * V).rolling(20).mean().values
    mk_a = mk.reindex(idx).ffill().fillna(False).values; tx_a = tx.reindex(idx).ffill().values
    out = []; last = -999
    for t in np.where(trans)[0]:
        if t - last < COOLDOWN or t < 260 or t + 1 >= n:
            continue
        last = t
        full = t >= 500
        a0 = t - 4; aH, aL = H[a0:t + 1].max(), L[a0:t + 1].min()
        top = old_top[t]
        # 突破日：A 之前 30 根內，第一根收盤站上 C 上緣
        cand = [i for i in range(t - 30, a0) if C[i] > top and C[i - 1] <= top]
        s_bo = cand[0] if cand else a0 - 1
        bH = H[s_bo:a0].max() if a0 > s_bo else H[s_bo]
        lo_i = t - 120 + int(np.argmin(L[t - 120:t - 20]))
        pk_i = max(0, lo_i - 300) + int(np.argmax(H[max(0, lo_i - 300):lo_i])) if lo_i > 5 else 0
        seg = C[pk_i:lo_i + 1]
        drop20 = min(seg[j + 20] / seg[j] - 1 for j in range(len(seg) - 20)) if len(seg) > 21 else seg[-1] / seg[0] - 1
        prev_lo = L[max(0, pk_i - 250):pk_i].min() if pk_i > 20 else np.nan
        Cvol = V[t - 120:t - 20].mean(); Bvol = V[s_bo:a0].mean() if a0 > s_bo else V[s_bo]
        f = dict(code=code, date=idx[t], close=C[t], raw=RC[t], amt20=amt20[t],
                 near5=near5[t], far5=far5[t], c_top=top,
                 b_ext=bH / top - 1, b_len=a0 - s_bo, bo_vol=V[s_bo:a0 + 1].max() / Cvol if Cvol > 0 else np.nan,
                 a_vol=V[a0:t + 1].mean() / Bvol if Bvol > 0 else np.nan,
                 a_width=(aH - aL) / C[t], a_low_vs_top=aL / top - 1, a_width_atr=(aH - aL) / atr[t],
                 dec_depth=L[lo_i] / H[pk_i] - 1, dec_len=lo_i - pk_i, dec_speed=(L[lo_i] / H[pk_i] - 1) / max(1, lo_i - pk_i),
                 dec_drop20=drop20, base_len=a0 - lo_i, base_dec_ratio=(a0 - lo_i) / max(1, lo_i - pk_i),
                 low_vs_prevlow=L[lo_i] / prev_lo if prev_lo and prev_lo > 0 else np.nan,
                 slope_now=slope[t], slope_min120=np.nanmin(slope[t - 120:t + 1]),
                 slope_rise=slope[t] - np.nanmin(slope[t - 120:t + 1]), align_run=run[t],
                 dist500=C[t] / H[max(0, t - 500):t + 1].max() - 1,
                 hi_ratio=H[t - 250:t - 20].max() / H[t - 500:t - 250].max() if full else np.nan,
                 lo_ratio=L[t - 250:t - 20].min() / L[t - 500:t - 250].min() if full else np.nan,
                 mkt=bool(mk_a[t]), rs60=(C[t] / C[t - 60] - 1) - (tx_a[t] / tx_a[t - 60] - 1),
                 ld=(e12[t] - e200[t]) / atr[t], z12=(C[t] - e12[t]) / atr[t])
        for nm, (x, y) in {'3m': (21, 80), '6m': (80, 150), '12m': (150, 250), '24m': (250, 500)}.items():
            d_, v_ = band_density(H, L, V, t, aH, aH * 1.10, x, y) if t >= y else (np.nan, np.nan)
            f[f'ovh_{nm}'] = d_; f[f'ovh_vol_{nm}'] = v_
        # C 區平整度：前 21～120 根的 90 分位高 / 10 分位低
        cq_hi, cq_lo = np.quantile(H[t - 120:t - 20], 0.9), np.quantile(L[t - 120:t - 20], 0.1)
        f['c_width'] = (cq_hi - cq_lo) / ((cq_hi + cq_lo) / 2)
        f['atr_pct'] = atr[t] / C[t]
        # 標籤：t+1 收盤進場、60 日內先漲 15% = 成功；三種失敗定義
        ent = C[t + 1]
        def lab(stop):
            for k in range(t + 1, min(n, t + 61)):
                if H[k] >= ent * 1.15: return 'win', k - t
                if C[k] < stop: return 'loss', k - t
            return 'none', np.nan
        res, days = lab(aL * 0.99)                                  # 原定義：跌破 A 低點
        f['label_c'], _ = lab(top * 0.97)                           # 跌破 C 上緣 3%
        f['label_atr'], _ = lab(ent - 2 * atr[t])                   # 跌破進場價 2 ATR
        fw = C[t + 1:t + 61]
        f.update(label=res, label_days=days, fwd20=C[min(n - 1, t + 21)] / ent - 1, fwd60=C[min(n - 1, t + 61)] / ent - 1,
                 mfe60=fw.max() / ent - 1 if len(fw) else np.nan, mae60=fw.min() / ent - 1 if len(fw) else np.nan,
                 complete=t + 61 < n)
        out.append(f)
    return out


if __name__ == '__main__':
    import time
    t0 = time.time()
    lst, tx_df, cal, S, _ = load_all()
    tx = tx_df.Close
    e60 = tx.ewm(span=60, adjust=False).mean(); mk = (e60 / e60.shift(20) - 1) > 0
    rows = []
    for i, (code, a) in enumerate(S.items(), 1):
        rows += scan_stock(code, a, mk, tx)
        if i % 300 == 0: print(f'{i}/{len(S)} 檔，事件 {len(rows)}，{time.time() - t0:.0f} 秒', flush=True)
    E = pd.DataFrame(rows)
    E['delisted'] = E.code.map(lst.set_index('code').market).eq('DL')
    E.to_pickle('/home/claude/bt/struct/events.pkl')
    print(f'完成：{len(E)} 個事件，{E.code.nunique()} 檔，{time.time() - t0:.0f} 秒')
