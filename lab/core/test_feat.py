# 步驟 2 檢查：core/feat.py 的指標
# A. 對照參考實作：從 lab/finlab/fl_small_report.py 原文抽出個股指標迴圈與 IAKF，原封不動執行在同一份輸入上，逐格比對。
# B. 獨立重算：不用 pandas，依規則文字逐日手算（ATR、EMA、布林、R20、S60、回撤比、EV2），驗證公式本身。
# C. 面板對齊：VALID、LAG20。
# 判準：最大絕對誤差 < 1e-6（相對於價格尺度：指標先除以收盤價或本身已無單位），布林事件完全一致。
# 用法：python lab/core/test_feat.py [AF|AF_C] [抽樣檔數]
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat

TOL = 1e-6
adj = sys.argv[1] if len(sys.argv) > 1 else 'AF'
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
P = load_panel(); F = load_feat(adj); M = P['meta']; T, N = P['C'].shape
VALID = np.asarray(F['VALID'])
nv = VALID.sum(0)
rng = np.random.default_rng(7)
fixed = [M.index[M.stock_id == s][0] for s in ('2330', '2317', '3008', '6130') if (M.stock_id == s).any()]
pick = sorted(set(fixed) | set(rng.choice(np.flatnonzero(nv >= 300), NS, replace=False)) | set(rng.choice(np.flatnonzero((nv >= 30) & (nv < 300)), 5, replace=False)))
res = {}
def rec(name, err):
    res[name] = max(res.get(name, 0.0), float(err))

# ---------- 參考實作的輸入（依 fl_small_report.py 第 25–32 行的做法，從未還原價組出還原價）
RC = np.asarray(P['C'], float)[:, pick]; ROP = np.asarray(P['O'], float)[:, pick]
RH = np.fmax(np.fmax(np.asarray(P['H'], float)[:, pick], ROP), RC); RL = np.fmin(np.fmin(np.asarray(P['L'], float)[:, pick], ROP), RC)
AFm = np.asarray(P[adj])[:, pick]
VV = VALID[:, pick]
mask = lambda x: np.where(VV, x, np.nan)
# 直接乘還原因子（不用 C/RC 反推：浮點誤差會讓同價高點分出高下，使回撤比選到不同高點）
C = mask(RC * AFm); H, L = mask(RH * AFm), mask(RL * AFm)
An = RC * np.asarray(P['V'], float)[:, pick]

# ---------- A. 參考實作原文
src = (LAB / 'finlab' / 'fl_small_report.py').read_text(encoding='utf-8').splitlines()
def block(start, end):
    i = next(k for k, s in enumerate(src) if s.startswith(start)); j = next(k for k in range(i, len(src)) if src[k].startswith(end))
    return '\n'.join(src[i:j + 1])
loop1 = block('for j in range(N_):', '    OLD[ii[249:], j] = True')
i2 = next(k for k, s in enumerate(src) if s.startswith('S60 = np.full'))
j2 = next(k for k in range(i2, len(src)) if src[k].startswith('    o_ = np.full(n_, np.nan)'))
loop2 = '\n'.join(src[i2 + 2:j2 + 1]).replace('Hn2', 'H').replace('Ln2', 'L')
iakf = block('def iakf_down', '    return pd.Series(dn, index=x.index)')
T_, N_ = C.shape; f32 = np.float32
ns = dict(np=np, pd=pd, swv=swv, T_=T_, N_=N_, Cn=C.astype(f32), Hn=H.astype(f32), Ln=L.astype(f32), VV=VV, An=An.astype(f32),
          H=H.astype(f32), L=L.astype(f32))
for k in ('ATR', 'NUM', 'AMT', 'R20', 'S60', 'RRT'): ns[k] = np.full((T_, N_), np.nan, f32)
for k in ('EV1', 'EV2', 'OLD'): ns[k] = np.zeros((T_, N_), bool)
exec(loop1, ns); exec(loop2, ns); exec(iakf, ns)
# 參考實作把輸入先轉 float32 → 兩邊同樣以 float32 輸入重跑我們的函式再比，避免把精度差異當成公式差異
from feat import stock_feats, iakf_down
for jj, j in enumerate(pick):
    ii = np.flatnonzero(VV[:, jj])
    if len(ii) < 30: continue
    c, h, l, a = (ns['Cn'][ii, jj].astype(float), ns['Hn'][ii, jj].astype(float), ns['Ln'][ii, jj].astype(float), ns['An'][ii, jj].astype(float))
    r = stock_feats(c, h, l, a); sc = np.nanmedian(c)
    for k, rk, s in (('ATR', 'ATR', sc), ('NUM', 'NUM', sc), ('AMT20', 'AMT', np.nanmedian(a)), ('R20', 'R20', 1), ('S60', 'S60', 1), ('RRT', 'RRT', 1)):
        x, y = r[k].astype(f32).astype(float), ns[rk][ii, jj].astype(float)
        both = np.isfinite(x) & np.isfinite(y); rec(f'A 參考 {k}', np.max(np.abs(x[both] - y[both])) / s if both.any() else 0)
        rec(f'A 參考 {k} 空值位置不同', (np.isfinite(x) != np.isfinite(y)).sum())
    for k in ('EV1', 'EV2', 'OLD'): rec(f'A 參考 {k} 不同格數', (r[k] != ns[k][ii, jj]).sum())
ixp = P['index']['TAIEX_P']
ref = ns['iakf_down'](ixp.dropna()).reindex(ixp.index).fillna(False).values
rec('A 參考 IAKF 不同天數', (ref != iakf_down(ixp.values)).sum())
rec('A 參考 IAKF 與快取不同天數', (ref != F['market']['IAKF_DN'].values).sum())

# ---------- 快取與函式一致（feat.build 的組裝：VALID、還原、夾住高低價）
for jj, j in enumerate(pick):
    ii = np.flatnonzero(VV[:, jj])
    for k, ref_ in (('CA', C), ('HA', H), ('LA', L)):
        x = np.asarray(F[k][ii, j], float); y = ref_[ii, jj]; rec(f'C 快取 {k}', np.max(np.abs(x / y - 1)) if len(ii) else 0)
    rec('C VALID 外有值', np.isfinite(np.asarray(F['CA'][~VALID[:, j], j])).sum())
    lag = np.asarray(F['LAG20'][ii, j]); exp = np.r_[np.full(min(20, len(ii)), -1), ii[:-20]] if len(ii) > 20 else np.full(len(ii), -1)
    rec('C LAG20 錯誤', (lag != exp).sum())
    if len(ii) >= 30:
        r = stock_feats(C[ii, jj], H[ii, jj], L[ii, jj], An[ii, jj])
        for k in ('ATR', 'NUM', 'S60', 'RRT', 'R20'):
            x = np.asarray(F[k][ii, j], float); y = r[k]; both = np.isfinite(y)
            s = np.nanmedian(C[ii, jj]) if k in ('ATR', 'NUM') else 1
            rec(f'C 快取 {k}', np.nanmax(np.abs(x[both] - y[both])) / s if both.any() else 0)

# ---------- B. 獨立重算（純迴圈，依規則文字）
def ema(x, alpha):
    o = np.empty(len(x)); o[0] = x[0]
    for t in range(1, len(x)): o[t] = alpha * x[t] + (1 - alpha) * o[t - 1]
    return o
for jj, j in enumerate(pick[:15]):
    ii = np.flatnonzero(VV[:, jj]); n = len(ii)
    if n < 300: continue
    c, h, l = C[ii, jj], H[ii, jj], L[ii, jj]; r = stock_feats(c, h, l, An[ii, jj]); sc = np.median(c)
    tr = np.array([np.nan] + [max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1])) for t in range(1, n)])
    atr = np.full(n, np.nan); atr[1] = tr[1]
    for t in range(2, n): atr[t] = atr[t - 1] + (tr[t] - atr[t - 1]) / 14          # Wilder
    rec('B 手算 ATR', np.nanmax(np.abs(atr[1:] - r['ATR'][1:])) / sc)
    num = ema(c, 2 / 13) - ema(c, 2 / 201); rec('B 手算 NUM（EMA12−EMA200）', np.max(np.abs(num - r['NUM'])) / sc)
    rec('B 手算 R20', np.nanmax(np.abs(c[20:] / c[:-20] - 1 - r['R20'][20:])))
    e2 = ema(c, 2 / 201); rec('B 手算 S60', np.nanmax(np.abs(e2[60:] / e2[:-60] - 1 - r['S60'][60:])))
    # 回撤比：近 60 日（含今天）最高價 pk 在 p；p 之前 60 日到 p 的最低價 lo；(pk − c)/(pk − lo)；波段 pk/lo−1 < 15% 或 p < 60 → 0
    rr = np.full(n, np.nan)
    for t in range(59, n):
        w = h[t - 59:t + 1]; p = t - 59 + int(np.argmax(w))
        if p < 60: rr[t] = 0.0; continue
        lo = np.min(l[p - 60:p + 1]); pk = h[p]
        rr[t] = (pk - c[t]) / (pk - lo) if pk / lo - 1 >= 0.15 and pk > lo else 0.0
    rec('B 手算 回撤比', np.nanmax(np.abs(rr[59:] - r['RRT'][59:])))
    # EV2：昨天的 (20 日高 − 20 日低)/ATR ≤ 到昨天為止前 250 日（≥150）的第 30 百分位，且今天收盤 > 前 20 日最高
    rg = np.full(n, np.nan)
    for t in range(19, n): rg[t] = (h[t - 19:t + 1].max() - l[t - 19:t + 1].min()) / r['ATR'][t]
    ev2 = np.zeros(n, bool)
    for t in range(21, n):
        hist = rg[max(0, t - 250):t]; hist = hist[np.isfinite(hist)]
        if len(hist) < 150 or not np.isfinite(rg[t - 1]): continue
        q = np.quantile(hist, 0.3)          # pandas rolling quantile 預設線性內插，同 numpy
        ev2[t] = rg[t - 1] <= q and c[t] > h[t - 20:t].max()
    rec('B 手算 EV2 不同格數', (ev2 != r['EV2']).sum())

# ---------- 報告
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
print(f'抽樣 {len(pick)} 檔（含 2330、2317、3008、6130 與 5 檔有效日 < 300 者），模式 {adj}')
bad = []
for k, v in res.items():
    lim = 0 if ('格數' in k or '天數' in k or '位置' in k or '錯誤' in k or '有值' in k) else TOL
    ok = v <= lim; bad += [] if ok else [k]
    print(f"{'通過' if ok else '不通過'}  {k}: {v:.3g}")
print('結論：' + ('全部通過' if not bad else f'{len(bad)} 項不通過'))
