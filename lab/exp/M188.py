# M-188：股池內分出左右尾的條件（奇數代碼；到 2024-11-13）。定義見 registry_M.md M-188。
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M165

END = pd.Timestamp('2024-11-13')
PARITY = int(os.environ.get('M188_PARITY', '1'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    par = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cf = pd.DataFrame(CA).ffill()
    C = Cf.to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    Rz = np.where(VALID & ~np.isnan(R), R, 0.0)
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, Rz, np.nan), axis=1))
    CX = np.cumsum(Rz - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    ATR = np.asarray(F['ATR'], np.float64); AMT = np.asarray(F['AMT20'], np.float64)
    EMA12 = Cf.ewm(span=12, adjust=False).mean().to_numpy()
    MX250 = Cf.rolling(250, min_periods=200).max().to_numpy()
    lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
    LU = (np.where(VALID, R, 0) >= 0.95 * lim).astype(float)
    LU10 = pd.DataFrame(LU).rolling(10, min_periods=1).sum().to_numpy()
    feats = {
        'R20 20 日漲幅': lambda tp: np.asarray(F['R20'])[tp],
        'S60 斜率': lambda tp: np.asarray(F['S60'])[tp],
        'R5 5 日漲幅': lambda tp: C[tp] / C[tp - 5] - 1,
        'R10 10 日漲幅': lambda tp: C[tp] / C[tp - 10] - 1,
        'EXT 乖離÷ATR': lambda tp: (C[tp] - EMA12[tp]) / ATR[tp],
        'ATR%': lambda tp: ATR[tp] / C[tp],
        '量增 AMT20÷60日前': lambda tp: AMT[tp] / AMT[tp - 60],
        '距250日高': lambda tp: C[tp] / MX250[tp],
    }
    rows = []
    for m, (t, tp) in sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0]):
        if dates[t] < pd.Timestamp('2005-01-01') or dates[t] > END or t + 61 >= T: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–24/11'
        Ut = U[t]; e60 = CX[t + 60] - CX[t]; q10, q90 = np.nanpercentile(e60[Ut], [10, 90])
        with np.errstate(all='ignore'): dd = np.nanmin(C[t + 1:t + 61], axis=0) / C[t] - 1
        fv = {k: f(tp) for k, f in feats.items()}
        for pname, pm in (('強勢池', pool[t]), ('營收池', A[t])):
            idx = np.nonzero(pm & Ut)[0]
            if len(idx) < 10: continue
            rk = {k: pd.Series(v[idx]).rank(pct=True).to_numpy() for k, v in fv.items()}
            for ii, j in enumerate(idx):
                if not par[j]: continue
                rows.append((era, pname, e60[j], e60[j] <= q10, e60[j] >= q90, dd[j] <= -0.20, LU10[tp, j] >= 1, *[rk[k][ii] for k in feats]))
    D = pd.DataFrame(rows, columns=['era', 'pool', 'e60', 'L', 'Rt', 'DD', '漲停≥1'] + list(feats))
    out = [f"# M-188 結果：股池內分出左右尾的條件（{'奇' if PARITY else '偶'}數代碼；60 日超額；左右尾 = 當日 U 的後／前 10%；DD = 60 日內最低收盤 ≤ −20%）\n",
           '格式：左尾｜右尾｜DD｜平均，Δ淨尾（百分點）＝旗標組（右−左）−同池其他股票（右−左）。左欄 2005–15、右欄 2016–24/11。\n']
    noms = []
    for pname in ('強勢池', '營收池'):
        out += [f'## {pname}\n', '| 條件 | 2005–15 旗標組 | Δ淨尾 | 2016–24/11 旗標組 | Δ淨尾 | 判定 |', '|---|---|---|---|---|---|']
        base = D[D.pool == pname]
        conds = [(f'{k} 最高 20%', base[k] > 0.8) for k in feats] + [(f'{k} 最低 20%', base[k] <= 0.2) for k in feats] + [('近 10 日漲停 ≥ 1', base['漲停≥1'])]
        for nm, flag in conds:
            cells, dn, lr = [], [], []
            for era in ('2005–15', '2016–24/11'):
                e = base.era == era; x, o = base[e & flag], base[e & ~flag]
                if len(x) < 50: cells += ['—', '—']; dn.append(np.nan); lr.append(np.nan); continue
                d = (x.Rt.mean() - x.L.mean()) - (o.Rt.mean() - o.L.mean())
                dn.append(d); lr.append(x.L.mean() / max(o.L.mean(), 1e-9))
                cells += [f"{x.L.mean():.0%}｜{x.Rt.mean():.0%}｜{x.DD.mean():.0%}｜{x.e60.mean():+.1%}（n={len(x)}）", f"{d * 100:+.1f}"]
            tag = ''
            if all(v <= -0.05 for v in dn) and all(v >= 1.3 for v in lr): tag = '**剔除候選**'
            elif all(v >= 0.05 for v in dn) and all(v <= 1.1 for v in lr): tag = '**偏好候選**'
            if tag: noms.append(f'{pname}：{nm} → {tag}')
            out.append(f"| {nm} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {tag} |")
        o2 = []
        for era in ('2005–15', '2016–24/11'):
            y = base[base.era == era]; o2.append(f"{era}：左尾 {y.L.mean():.0%}｜右尾 {y.Rt.mean():.0%}｜DD {y.DD.mean():.0%}｜平均 {y.e60.mean():+.1%}（n={len(y)}）")
        out += ['', f'{pname}整體：' + '；'.join(o2), '']
    out += ['## 提名\n'] + ([f'- {s}' for s in noms] or ['- 無'])
    return dict(summary='\n'.join(out), headline=f'提名 {len(noms)} 項', n_variants=0)
