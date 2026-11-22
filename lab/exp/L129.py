# L-129：營收訊號組合加入大盤週 EMA 閘門（停新倉）與空頭保本。判準見 registry_L.md L-129。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, evalu as E
from M122 import fin_factors
import M165

FEE, TAX, SLIP = 0.00085, 0.003, 0.0015; HMAX = 120
VARS = {'G0 基準＝L-127 B2': ('atr', 40, 0, False), 'G1 EMA21＋停新倉＋保本（主）': ('atr', 40, 21, True), 'G2 EMA16': ('atr', 40, 16, True),
        'G3 EMA26': ('atr', 40, 26, True), 'G4 EMA21 只停新倉': ('atr', 40, 21, False)}
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'),
        ('2016–26', '2016-01-01', '2026-12-31'), ('全期', '2005-01-01', '2026-12-31'))
N_DSR = 597
from M144 import weekly_gate


def seg(r, lo, hi):
    r = r.loc[lo:hi]; n = len(r); eq = (1 + r).cumprod(); ann = eq.iloc[-1] ** (245 / n) - 1; dd = (eq / eq.cummax() - 1).min()
    return ann, dd, (ann / -dd if dd < 0 else np.nan)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; del S
    if Tr.shape[1] != N: full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    ATRP = np.asarray(F['ATR'], np.float64) / CA
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    out_m = np.vstack([np.zeros((1, N), bool), (U & ~pool)[:-1]])
    with np.errstate(all='ignore'): ewo = pd.Series(np.nan_to_num(np.nanmean(np.where(out_m, R, np.nan), axis=1)), index=dates)
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    GATE = {n: weekly_gate(ix, dates, n) for n in (16, 21, 26)}
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    SIG = {}
    for m, (t, _) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2004-07-01') or t + 2 >= T or not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = np.where(qj[mem] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[mem], 0, None)], np.nan) if m in Q5.index else np.full(len(mem), np.nan)
        fin = np.isfinite(q); drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        s = mem[~drop]; s = s[VALID[t + 1, s] & np.isfinite(OA[t + 1, s]) & (OA[t + 1, s] > 0)]
        SIG[t + 1] = (t, s)

    def run(xm, vc, gn, be):
        cash = 1.0; pos = {}; eq = np.full(T, np.nan); trades = []; nh = np.zeros(T); cs = np.zeros(T); gd = np.zeros(T, bool)
        t_start = min(SIG)
        for t in range(t_start, T):
            neg = bool(gn) and bool(GATE[gn][t - 1]); gd[t] = neg
            if t in SIG and not neg:                             # 開盤買進（閘門用前一日收盤狀態）
                ts, s = SIG[t]; s = [j for j in s if j not in pos]
                if s and cash > 1e-9:
                    a = cash / len(s)
                    for j in s:
                        px = OA[t, j] * (1 + SLIP); d = 4 * ATRP[t - 1, j]
                        d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
                        pos[j] = dict(sh=a * (1 - FEE) / px, px=px, hi=px, d=d, t0=t, ts=ts)
                    cash = 0.0
            sell = {}
            for j, p in pos.items():
                c = Cf[t, j]
                if not np.isfinite(c): continue
                p['hi'] = max(p['hi'], c); age = t - p['t0'] + 1
                gneg = bool(gn) and bool(GATE[gn][t])
                if be:
                    if not gneg: p['be'] = False
                    elif c > p['px']: p['be'] = True
                if be and p.get('be') and c < p['px'] * 1.005: sell[j] = '空頭保本'
                elif xm == 'hard' and c < p['px'] * 0.85: sell[j] = '硬停損'
                elif xm == 'atr' and c < p['hi'] * (1 - p['d']): sell[j] = 'ATR 移動停損'
                elif age >= HMAX: sell[j] = '120日到期'
            if vc:                                              # V2：訊號日＋vc 的收盤
                coh = {}
                for j, p in pos.items():
                    if j not in sell and t == p['ts'] + vc: coh.setdefault(p['ts'], []).append(j)
                for ts, jj in coh.items():
                    if len(jj) < 5: continue
                    v = np.array([Tr[t, j] - Tr[ts, j] for j in jj]); ok = np.isfinite(v)
                    if ok.sum() < 5: continue
                    thr = np.nanpercentile(v[ok], 20)
                    for j, x in zip(jj, v):
                        if np.isfinite(x) and x <= thr: sell[j] = 'V2 提前出場'
            for j, why in sell.items():
                p = pos.pop(j); v = p['sh'] * Cf[t, j] * (1 - FEE - TAX - SLIP); cash += v
                trades.append((t, why, v / (p['sh'] * p['px'] / (1 - FEE)) - 1, t - p['t0'] + 1))
            val = cash + sum(p['sh'] * Cf[t, j] for j, p in pos.items() if np.isfinite(Cf[t, j]))
            eq[t] = val; nh[t] = len(pos); cs[t] = cash / val if val > 0 else 1
        r = pd.Series(eq, index=dates).dropna().pct_change().fillna(0)
        X = pd.DataFrame(trades, columns=['t', 'why', 'ret', 'hold']); X['date'] = dates[X.t]
        return r, X, pd.Series(nh, index=dates), pd.Series(cs, index=dates), pd.Series(gd, index=dates)

    OUT = {nm: run(*v) for nm, v in VARS.items()}
    L = ['# L-129 結果：營收訊號組合＋大盤週 EMA 閘門與空頭保本（現金再投入研究用組合）\n', '## 各段（年化／最大回撤／Calmar）\n',
         '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|---|']
    for nm, (r, *_ ) in list(OUT.items()) + [('加權報酬指數 TR', (tr,)), ('等權池外股票', (ewo,))]:
        L.append(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*seg(r, lo, hi)) for _, lo, hi in SEGS) + ' |')
    L += ['', '## 相對超額（年化，只報告）\n', '| 版本 | 相對 TR 05–15 | 相對 TR 16–26 | 相對等權池外 05–15 | 相對等權池外 16–26 |', '|---|---|---|---|---|']
    for nm, (r, *_ ) in OUT.items():
        f = lambda b, lo, hi: seg(r, lo, hi)[0] - seg(b, lo, hi)[0]
        L.append(f"| {nm} | {f(tr, '2005-01-01', '2015-12-31'):+.1%} | {f(tr, '2016-01-01', '2026-12-31'):+.1%} | {f(ewo, '2005-01-01', '2015-12-31'):+.1%} | {f(ewo, '2016-01-01', '2026-12-31'):+.1%} |")
    L += ['', '## 交易\n', '| 版本 | 每年筆數 | 平均持有日 | 每筆平均 | 勝率 | 出場原因 | 平均持股檔數 | 平均現金比例 | 閘門為負天數 | 保本出場每筆平均 |', '|---|---|---|---|---|---|---|---|---|---|']
    for nm, (r, X, nh, cs, gd) in OUT.items():
        X = X[X.date >= '2005-01-01']; yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
        wy = '、'.join(f'{k} {v:.0%}' for k, v in X.why.value_counts(normalize=True).items())
        bx = X[X.why == '空頭保本']
        L.append(f"| {nm} | {len(X) / yrs:.0f} | {X.hold.mean():.0f} | {X.ret.mean():+.2%} | {(X.ret > 0).mean():.0%} | {wy} | {nh.loc['2005':].mean():.0f} | {cs.loc['2005':].mean():.0%} | {gd.loc['2005':].mean():.0%} | {(f'{bx.ret.mean():+.2%}（{len(bx)} 筆）' if len(bx) else '—')} |")
    b0 = OUT['G0 基準＝L-127 B2'][0]
    L += ['', '## 判準（相對 G0；G1 判定；G2、G3 高原；G4 只報告）\n', '| 版本 | Calmar 16–20 | 21–26 | bootstrap P（16–26） | 05–15 回撤變化 | DSR | 結論 |', '|---|---|---|---|---|---|---|']
    var_sr = np.var([E.met(o[0].loc['2005':])[3] / np.sqrt(252) for o in OUT.values()])
    for nm, (r, *_ ) in OUT.items():
        if nm.startswith('G0'): continue
        c1 = seg(r, '2016-01-01', '2020-12-31')[2] - seg(b0, '2016-01-01', '2020-12-31')[2]
        c2 = seg(r, '2021-01-01', '2026-12-31')[2] - seg(b0, '2021-01-01', '2026-12-31')[2]
        pb = E.boot_p(r, b0); dd = seg(r, '2005-01-01', '2015-12-31')[1] - seg(b0, '2005-01-01', '2015-12-31')[1]
        try: dsr = E.dsr(r.loc['2005':], N_DSR, var_sr)
        except Exception: dsr = np.nan
        judged = nm.startswith('G1')
        ok = c1 > 0 and c2 > 0 and pb >= 0.8 and dd >= -0.05
        L.append(f"| {nm} | {c1:+.2f} | {c2:+.2f} | {pb:.2f} | {dd * 100:+.1f}pt | {dsr:.2f} | {('**通過**' if ok else '不通過') if judged else '只報告'} |")
    L += ['', '## 每年報酬\n', '| 年 | ' + ' | '.join(k.split(' ')[0] for k in OUT) + ' | TR | 等權池外 |', '|' + '---|' * (len(OUT) + 3)]
    yr = lambda r: (1 + r.loc['2005':]).groupby(r.loc['2005':].index.year).prod() - 1
    Y = pd.DataFrame({k.split(' ')[0]: yr(v[0]) for k, v in OUT.items()}); Y['TR'] = yr(tr); Y['等權池外'] = yr(ewo)
    for y, row in Y.iterrows(): L.append(f'| {y} | ' + ' | '.join(f'{x:+.1%}' for x in row) + ' |')
    r2 = OUT['G1 EMA21＋停新倉＋保本（主）'][0]
    hl = f"G1 Calmar {seg(r2, '2005-01-01', '2015-12-31')[2]:.2f}／{seg(r2, '2016-01-01', '2020-12-31')[2]:.2f}／{seg(r2, '2021-01-01', '2026-12-31')[2]:.2f} vs G0 {seg(b0, '2005-01-01', '2015-12-31')[2]:.2f}／{seg(b0, '2016-01-01', '2020-12-31')[2]:.2f}／{seg(b0, '2021-01-01', '2026-12-31')[2]:.2f}"
    return dict(summary='\n'.join(L), headline=hl, n_variants=4, series={k.split(' ')[0]: v[0] for k, v in OUT.items()})
