# 個股指標（以個股自身有效交易日計算）→ lab/cache/feat/<adj>/*.npy
# 公式依 lab/finlab/fl_small_report.py（2026-11-15 定版的實作）；對照測試見 lab/core/test_feat.py。
# 用法：python lab/core/feat.py [AF|AF_C]      AF＝完整還原（修正模式），AF_C＝相容模式（偏誤 ②）
#       from lab.core.feat import load_feat
#
# 有效日 VALID ＝ 有成交、未停牌、收盤 > 0、非興櫃（panel 的 TRD & ~EMG）。
# 只在 VALID 日計算；其他日子為 NaN／False。橫斷面（全市場分位數）的東西不在這裡，在 signals.py。
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel, peak_mem_mb

OUT = LAB / 'cache' / 'feat'
F32 = ['CA', 'OA', 'HA', 'LA', 'ATR', 'NUM', 'AMT20', 'AMT20V', 'R20', 'S60', 'RRT']
# AMT20：未還原收盤 × 股數（參考實作，相容模式）；AMT20V：成交金額 value（修正模式，L-113 補充）
BOOL = ['VALID', 'EV1', 'EV2', 'OLD']
I32 = ['LAG20']     # 同一檔股票往前第 20 個有效日的列號（沒有則 −1），給 atr_ok 用


def stock_feats(c, h, l, a):
    """單一股票、只含有效日的序列（float64）。回傳 dict（長度相同的陣列）。"""
    n = len(c); s = pd.DataFrame({'C': c, 'H': h, 'L': l, 'A': a})
    pc = s.C.shift(1)
    tr = np.maximum(s.H - s.L, np.maximum((s.H - pc).abs(), (s.L - pc).abs()))   # 第一天 pc 為空 → TR 為空
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()                                # Wilder ATR(14)
    m20 = s.C.rolling(20).mean(); sd20 = s.C.rolling(20).std()                     # 布林 20 日、樣本標準差
    up = m20 + 2 * sd20; bw = 4 * sd20 / m20                                       # 上軌、帶寬
    hh20 = s.H.rolling(20).max(); rng = (hh20 - s.L.rolling(20).min()) / atr      # 20 日區間 ÷ ATR
    o = dict(ATR=atr.values,
             NUM=(s.C.ewm(span=12, adjust=False).mean() - s.C.ewm(span=200, adjust=False).mean()).values,
             AMT20=s.A.rolling(20, min_periods=15).mean().values,
             R20=(s.C / s.C.shift(20) - 1).values,
             # 布林收斂後突破：近 10 日（到昨天）帶寬最小值 ≤ 前 120 日（到昨天）第 20 百分位，今天收盤首次站上上軌
             EV1=((bw.rolling(10).min().shift(1) <= bw.rolling(120, min_periods=100).quantile(0.2).shift(1))
                  & (s.C > up) & (pc <= up.shift(1))).values,
             # 整理後突破 20 日高：昨天的區間/ATR ≤ 前 250 日（到昨天）第 30 百分位，今天收盤 > 前 20 日最高
             EV2=((rng.shift(1) <= rng.rolling(250, min_periods=150).quantile(0.3).shift(1)) & (s.C > hh20.shift(1))).values,
             OLD=np.arange(n) >= 249)                                               # 已有 250 個有效日
    # EMA200 近 60 日斜率、60 日波段回撤比（需 ≥ 130 個有效日）
    S60 = np.full(n, np.nan); RRT = np.full(n, np.nan)
    if n >= 130:
        e2 = s.C.ewm(span=200, adjust=False).mean().values
        S60[60:] = e2[60:] / e2[:-60] - 1
        h_ = np.where(np.isfinite(h), h, -np.inf); l_ = np.where(np.isfinite(l), l, np.inf)
        tt = np.arange(59, n)
        p_ = tt - 59 + swv(h_, 60).argmax(1)                  # 近 60 日最高價的位置（同價取最早）
        lw = swv(l_, 61).argmin(1); ok = p_ >= 60              # 高點前 60 日內（含高點）的最低價
        q_ = np.where(ok, p_ - 60 + lw[np.where(ok, p_ - 60, 0)], 0)
        pk, lo, cc = h_[p_], l_[q_], c[tt]
        with np.errstate(all='ignore'):
            rr = (pk - cc) / (pk - lo); sw = pk / lo - 1
        RRT[tt] = np.where(ok & (sw >= 0.15) & np.isfinite(rr), rr, 0.0)   # 波段漲幅 < 15% 視為無波段（0）
    o['S60'] = S60; o['RRT'] = RRT
    return o


def dvs_own(CA, VALID, W=60):
    """下檔波動占比 DVS＝√Σ(r²; r<0) ÷ √Σr²，r＝個股自身有效日的還原日報酬，近 W 個有效日（至少 0.75W）。L-108P。"""
    CA = np.asarray(CA); VALID = np.asarray(VALID); T, N = CA.shape; out = np.full((T, N), np.nan, np.float32)
    mp = int(np.ceil(0.75 * W))
    for j in range(N):
        ii = np.flatnonzero(VALID[:, j])
        if len(ii) <= mp: continue
        c = CA[ii, j].astype(float); r = np.r_[np.nan, c[1:] / c[:-1] - 1]
        s = pd.Series(r); neg = (s.where(s < 0, 0.0) ** 2).where(s.notna())
        out[ii, j] = np.sqrt(neg.rolling(W, min_periods=mp).sum() / (s ** 2).rolling(W, min_periods=mp).sum()).values
    return out


def iakf_down(p_, smooth=250, nlook=10, thr=2.2, rej=0.05, decay=0.9):
    """大盤向下異常（IAKF 250/2.2）。輸入指數序列（可含 NaN），回傳 bool 陣列。"""
    p_ = np.asarray(p_, float); ratio = (2.0 / (smooth + 1)) ** 2; a = 2.0 / (nlook + 1); thr2 = thr ** 2
    dn = np.zeros(len(p_), bool); started = False
    for t, p in enumerate(p_):
        if np.isnan(p): continue
        if not started:
            est, S, mS, R, calm, ev = p, 1e-4, 0.0, 1e-4, 1e-4, 1e-4; Q = 1e-4 * ratio; started = True
        pv = ev + Q; sur = p - est; cs = sur - mS; an = cs ** 2 / S > thr2
        mS = (1 - a) * mS + a * sur; S = (1 - a) * S + a * cs ** 2
        R = rej * cs ** 2 + R if an else (1 - decay) * S + decay * R
        calm = (1 - decay) * S + decay * calm
        K = pv / (pv + R); est = est + K * sur; ev = (1 - K) * pv; Q = S * ratio
        dn[t] = an and cs < 0
    return dn


def compute(P, adj='AF', out=None):
    """P：面板 dict（load_panel() 或記憶體中的子集）。out＝None 時回傳記憶體陣列，否則寫入該資料夾（記憶體映射）。"""
    T, N = P['C'].shape
    if out is None:
        A = {f: np.full((T, N), np.nan, np.float32) for f in F32}
        A.update({f: np.zeros((T, N), bool) for f in BOOL}); A['LAG20'] = np.full((T, N), -1, np.int32)
    else:
        A = {f: np.lib.format.open_memmap(out / f'{f}.npy', 'w+', np.float32, (T, N)) for f in F32}
        for f in F32: A[f][:] = np.nan
        for f in BOOL: A[f] = np.lib.format.open_memmap(out / f'{f}.npy', 'w+', np.bool_, (T, N))
        A['LAG20'] = np.lib.format.open_memmap(out / 'LAG20.npy', 'w+', np.int32, (T, N)); A['LAG20'][:] = -1
    VALID = np.asarray(P['TRD']) & ~np.asarray(P['EMG']); A['VALID'][:] = VALID
    AF = P[adj]
    for j in range(N):
        ii = np.flatnonzero(VALID[:, j])
        if len(ii) == 0: continue
        f = np.asarray(AF[ii, j], float)
        rc = np.asarray(P['C'][ii, j], float); ro = np.asarray(P['O'][ii, j], float)
        rh = np.fmax(np.fmax(np.asarray(P['H'][ii, j], float), ro), rc)      # 高低價夾住開收（同參考實作）
        rl = np.fmin(np.fmin(np.asarray(P['L'][ii, j], float), ro), rc)
        c, o, h, l = rc * f, ro * f, rh * f, rl * f
        A['CA'][ii, j], A['OA'][ii, j], A['HA'][ii, j], A['LA'][ii, j] = c, o, h, l
        if len(ii) > 20: A['LAG20'][ii[20:], j] = ii[:-20]
        if len(ii) < 30: continue                                             # 同參考實作：少於 30 個有效日不算指標
        r = stock_feats(c, h, l, rc * np.asarray(P['V'][ii, j], float))
        A['AMT20V'][ii, j] = pd.Series(np.asarray(P['AMT'][ii, j], float)).rolling(20, min_periods=15).mean().values
        for k in ('ATR', 'NUM', 'AMT20', 'R20', 'S60', 'RRT'): A[k][ii, j] = r[k]
        for k in ('EV1', 'EV2', 'OLD'): A[k][ii, j] = r[k]
    A['market'] = pd.DataFrame({'IAKF_DN': iakf_down(P['index']['TAIEX_P'].values)}, index=P['dates'])
    return A


def build(adj='AF'):
    """adj：AF 或 AF_C。兩種成交額（AMT20、AMT20V）都算。"""
    t0 = time.time(); P = load_panel()
    out = OUT / adj; out.mkdir(parents=True, exist_ok=True)
    A = compute(P, adj, out); VALID = np.asarray(A['VALID'])
    for f in A:
        if f != 'market': A[f].flush()
    A['market'].reset_index(names='date').to_parquet(out / 'market.parquet', index=False)
    info = dict(adj=adj, panel=P['info']['built'], tag=P['info']['tag'], seconds=round(time.time() - t0),
                peak_mb=round(peak_mem_mb()), valid_cells=int(VALID.sum()))
    (out / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


def load_feat(adj='AF', fields=None):
    d = OUT / adj
    out = {f: np.load(d / f'{f}.npy', mmap_mode='r') for f in (fields or F32 + BOOL + I32)}
    out['market'] = pd.read_parquet(d / 'market.parquet').set_index('date')
    out['info'] = json.loads((d / 'info.json').read_text(encoding='utf-8'))
    return out


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    for adj in (sys.argv[1:] or ['AF', 'AF_C']):
        print(json.dumps(build(adj), ensure_ascii=False), flush=True)
