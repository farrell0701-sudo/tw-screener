# M-192：X1 候選的三個進場狀態（太抖／太遠／太鬆）。定義見 registry_M.md M-192。
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals

END = pd.Timestamp('2024-11-13')
PARITY = int(os.environ.get('M192_PARITY', '1'))
THR = json.loads(os.environ['M192_THR']) if os.environ.get('M192_THR') else None
ERAS = ('2005–15', '2016–24/11')
VARS = ('太抖 ATR%排名', '太遠 EXT50', '太鬆 整理緊度')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']
    par = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); RKX = full(S['RKX'], np.nan); below = S['daily']['below'].to_numpy(bool); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cdf = pd.DataFrame(CA).ffill(); C = Cdf.to_numpy(); O = np.asarray(F['OA'], np.float64)
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    Rz = np.where(VALID & ~np.isnan(R), R, 0.0)
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, Rz, np.nan), axis=1))
    CX = np.cumsum(Rz - ew[:, None], axis=0)
    ATR = np.asarray(F['ATR'], np.float64); ATRP = ATR / C
    EMA50 = Cdf.ewm(span=50, adjust=False).mean().to_numpy()
    HA, LA = np.asarray(F['HA'], np.float64), np.asarray(F['LA'], np.float64)
    RNG20 = pd.DataFrame(HA).rolling(20, min_periods=15).max().to_numpy() - pd.DataFrame(LA).rolling(20, min_periods=15).min().to_numpy()
    with np.errstate(all='ignore'): ar = pd.DataFrame(np.where(U, ATRP, np.nan)).rank(axis=1, pct=True).to_numpy()
    tend = int(dates.searchsorted(END, side='right')) - 1
    rows = []
    s0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
    for s in range(s0, tend - 1, 5):
        if below[s]: continue
        for j in np.nonzero((RKX[s] <= 12) & par)[0]:
            t = s + 1; px = O[t, j]
            if not np.isfinite(px) or px <= 0: continue
            d = ATRP[s, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
            hi = px; ex = None
            for k in range(t, min(t + 250, tend)):
                cc = CA[k, j]
                if not np.isfinite(cc): continue
                hi = max(hi, cc)
                if cc < hi * (1 - d) or (hi >= px * 1.06 and cc < px * 1.005) or k - t + 1 >= 250:
                    o = O[k + 1, j]; ex = (o if np.isfinite(o) else cc, k + 1 - t); break
            if ex is None: continue
            e60 = CX[min(s + 60, T - 1), j] - CX[s, j]
            rows.append((ERAS[0] if dates[s] <= pd.Timestamp('2015-12-31') else ERAS[1], ex[0] / px - 1, ex[1], e60,
                         ar[s, j], (C[s, j] - EMA50[s, j]) / ATR[s, j], RNG20[s, j] / ATR[s, j]))
    D = pd.DataFrame(rows, columns=['era', 'ret', 'hold', 'e60'] + list(VARS))
    thr = THR or {v: {str(q): float(D[v].quantile(q / 100)) for q in (70, 80, 90)} for v in VARS}
    def st(x):
        return dict(mu=x.ret.mean(), win=(x.ret > 0).mean(), loss=(x.ret <= -0.10).mean(), big=(x.ret >= 0.30).mean(), hold=x.hold.mean(), e60=x.e60.mean(), n=len(x))
    def fmt(s):
        return f"{s['mu']:+.1%}｜{s['win']:.0%}｜{s['loss']:.0%}｜{s['big']:.0%}｜{s['hold']:.0f}｜{s['e60']:+.1%}（n={s['n']}）"
    out = [f"# M-192 結果：X1 候選的三個進場狀態（{'奇' if PARITY else '偶'}數代碼；名次 ≤ 12、閘門開、每 5 日取樣；到 2024-11-13）\n",
           '格式：模擬交易平均｜勝率｜≤−10%｜≥+30%｜持有天數｜60 日超額（n）。\n',
           '門檻（奇數代碼第 70／80／90 百分位，確認集沿用）：' + '；'.join(f"{v} {thr[v]['70']:.3g}／{thr[v]['80']:.3g}／{thr[v]['90']:.3g}" for v in VARS) + '\n',
           f"全體：{fmt(st(D[D.era == ERAS[0]]))} ｜ {fmt(st(D[D.era == ERAS[1]]))}\n",
           '| 狀態（≥ 第 80 百分位） | 2005–15 旗標｜其他 | 2016–24/11 旗標｜其他 | 判定 |', '|---|---|---|---|']
    flags = {v: D[v] >= thr[v]['80'] for v in VARS}; verdict = {}
    for v in VARS:
        f = flags[v]; cells = []; ex_ok = dn_ok = True
        for e in ERAS:
            a, b = st(D[f & (D.era == e)]), st(D[~f & (D.era == e)])
            cells.append(f"{fmt(a)}<br>{fmt(b)}")
            ex_ok &= (a['mu'] <= b['mu'] - 0.01) and (a['e60'] <= b['e60'])
            dn_ok &= a['loss'] >= 1.5 * b['loss']
        verdict[v] = '**剔除候選**' if ex_ok else ('**降權候選**' if dn_ok else '—')
        out.append(f"| {v} | {cells[0]} | {cells[1]} | {verdict[v]} |")
    out += ['', '## 高原檢查（旗標組平均 − 其他平均，百分點；2005–15｜2016–24/11）\n', '| 狀態 | ≥ P70 | ≥ P80 | ≥ P90 |', '|---|---|---|---|']
    for v in VARS:
        cs = []
        for q in ('70', '80', '90'):
            f = D[v] >= thr[v][q]
            cs.append('｜'.join(f"{(D[f & (D.era == e)].ret.mean() - D[~f & (D.era == e)].ret.mean()) * 100:+.1f}" for e in ERAS))
        out.append(f"| {v} | {cs[0]} | {cs[1]} | {cs[2]} |")
    any_ = flags[VARS[0]] | flags[VARS[1]] | flags[VARS[2]]
    out += ['', '## 重疊與合併\n', f"旗標比例：" + '；'.join(f"{v} {flags[v].mean():.0%}" for v in VARS) + f"；任一 {any_.mean():.0%}",
            '相關（旗標）：' + '；'.join(f"{VARS[a][:2]}×{VARS[b][:2]} {np.corrcoef(flags[VARS[a]], flags[VARS[b]])[0, 1]:.2f}" for a, b in ((0, 1), (0, 2), (1, 2))),
            f"任一旗標：{fmt(st(D[any_ & (D.era == ERAS[0])]))} ｜ {fmt(st(D[any_ & (D.era == ERAS[1])]))}",
            f"都沒有：{fmt(st(D[~any_ & (D.era == ERAS[0])]))} ｜ {fmt(st(D[~any_ & (D.era == ERAS[1])]))}", '',
            '## 判定\n'] + [f"- {v}：{verdict[v]}" for v in VARS]
    (LAB / 'results' / 'M192_thr.json').write_text(json.dumps(thr, ensure_ascii=False)) if THR is None else None
    nom = [v for v in VARS if verdict[v] != '—']
    return dict(summary='\n'.join(out), headline=('提名：' + '、'.join(f"{v}{verdict[v]}" for v in nom)) if nom else '無提名', n_variants=0 if THR is None else len(VARS))
