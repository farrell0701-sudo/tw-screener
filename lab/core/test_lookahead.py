# 步驟 4 檢查之三：前視測試
# 取 400 檔股票，把 D 日之後的價格、成交量、指數隨機打亂（並隨機製造停牌），整條流程（指標 → 訊號 → 模擬）重跑一次。
# 判準：D 日以前（含）的指標、X1 排名、大盤狀態、淨值、已發生的交易完全相同。
# 已知、已登記的例外：V4.1 排名在 D 日當天可以不同（參考實作看次日開盤是否漲停，使用者同意保留：預選名單遞補，L-113 補充）。
# 用法：python lab/core/test_lookahead.py [D 日，預設 2015-06-30]
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
import feat, signals, sim

try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
D = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2015-06-30')
P = load_panel(); dates = P['dates']; tD = int(dates.searchsorted(D, side='right')) - 1
rng = np.random.default_rng(11)
M = P['meta']; nv = np.asarray(P['TRD']).sum(0)
cols = np.sort(np.r_[rng.choice(np.flatnonzero(nv > 1000), 380, replace=False), rng.choice(np.flatnonzero((nv > 100) & (nv <= 1000)), 20, replace=False)])
FIELDS = ['C', 'O', 'H', 'L', 'V', 'AMT', 'LIMU', 'LIMD', 'AF', 'AF_C', 'EX', 'TRD', 'SUSP', 'EMG', 'NOLIM', 'LUL', 'LDL', 'OSUS', 'DEF', 'DISP', 'MISS']


def subset():
    Q = {k: np.asarray(P[k])[:, cols].copy() for k in FIELDS}
    Q.update(dates=dates, meta=M.iloc[cols].reset_index(drop=True), index=P['index'].copy(), info=P['info'])
    return Q


def perturb(Q):
    T, N = Q['C'].shape; k = slice(tD + 1, None); n = T - tD - 1
    f = np.exp(np.cumsum(rng.normal(0, 0.03, (n, N)), axis=0)).astype(np.float32)
    for x in ('C', 'O', 'H', 'L'): Q[x][k] *= f
    Q['V'][k] *= rng.uniform(0.3, 3, (n, N)).astype(np.float32); Q['AMT'][k] *= rng.uniform(0.3, 3, (n, N)).astype(np.float32)
    Q['TRD'][k] &= rng.random((n, N)) > 0.03                         # 隨機停牌
    Q['LUL'][k] = rng.random((n, N)) < 0.02; Q['LDL'][k] = rng.random((n, N)) < 0.02; Q['OSUS'][k] = rng.random((n, N)) < 0.01
    ix = Q['index']; m = ix.index > D
    ix.loc[m, 'TAIEX_P'] = ix.loc[m, 'TAIEX_P'] * np.exp(np.cumsum(rng.normal(0, 0.015, m.sum())))
    return Q


def pipeline(Q, mode):
    md = signals.MODES[mode]
    F = feat.compute(Q, 'AF_C' if md['bias2'] else 'AF')
    S = signals.compute(mode, P=Q, F=F)
    c = sim.make_ctx(mode, Q, F, S)
    runs = {(k, i): (sim.runX if k == 'X' else sim.runV)(c, off=i, seed=1000 + i, n=6 if k == 'X' else 4) for k in ('X', 'V') for i in (0, 1)}
    return F, S, runs


def eqnan(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype == bool: return int((a != b).sum())
    return int(((np.isnan(a) != np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 0)).sum())


res = {}
for mode in ('compat', 'fix'):
    F0, S0, R0 = pipeline(subset(), mode)
    F1, S1, R1 = pipeline(perturb(subset()), mode)
    w = slice(0, tD + 1)
    for k in ('CA', 'OA', 'ATR', 'NUM', 'AMT20', 'AMT20V', 'R20', 'S60', 'RRT', 'EV1', 'EV2', 'OLD', 'VALID', 'LAG20'):
        res[f'{mode} 指標 {k}'] = eqnan(F0[k][w], F1[k][w])
    res[f'{mode} IAKF'] = int((F0['market']['IAKF_DN'].values[w] != F1['market']['IAKF_DN'].values[w]).sum())
    res[f'{mode} X1 排名 RKX'] = eqnan(S0['RKX'][w], S1['RKX'][w])
    res[f'{mode} V4.1 排名 RKV（D 日前一天為止）'] = eqnan(S0['RKV'][:tD], S1['RKV'][:tD])
    dv = eqnan(S0['RKV'][tD], S1['RKV'][tD])
    for c_ in ('below', 'bear_mode', 'off', 'n_pool', 'n_signal', 'floor', 'amt_q'):
        res[f'{mode} 每日 {c_}'] = eqnan(S0['daily'][c_].values[w].astype(float), S1['daily'][c_].values[w].astype(float))
    for key in R0:
        r0, r1 = R0[key], R1[key]
        res[f'{mode} {key[0]} 路徑 {key[1]} 淨值'] = eqnan(r0[0].loc[:D].values, r1[0].loc[:D].values)
        t0 = [x for x in r0[1] if x[2] <= D]; t1 = [x for x in r1[1] if x[2] <= D]
        res[f'{mode} {key[0]} 路徑 {key[1]} 已平倉交易'] = 0 if [x[:8] for x in t0] == [x[:8] for x in t1] else 1
        e0 = sorted((x[0], x[1]) for x in r0[1] if x[1] <= D); e1 = sorted((x[0], x[1]) for x in r1[1] if x[1] <= D)
        res[f'{mode} {key[0]} 路徑 {key[1]} 進場'] = 0 if e0 == e1 else 1
    print(f'{mode}：V4.1 排名在 D 日當天不同的格數 {dv}（已登記的例外）；D 日之後 X1 交易筆數 {len([x for x in R0[("X", 0)][1] if x[2] > D])} → {len([x for x in R1[("X", 0)][1] if x[2] > D])}（確認打亂確實改變了之後的結果）', flush=True)

bad = {k: v for k, v in res.items() if v}
for k, v in res.items(): print(f"{'通過' if not v else '不通過'}  {k}：不同 {v}")
print(f'D 日 {dates[tD].date()}｜{len(cols)} 檔｜結論：' + ('全部通過' if not bad else f'{len(bad)} 項不通過'))
