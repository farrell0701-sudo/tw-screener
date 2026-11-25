# M-188：大盤跌破多空點時，依個股 20 日波動 V20 與 T_ld 20 日變化 ΔT 分別處理（奇數代碼）。定義見 registry_M.md M-188。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
import M182, M185, M186
from panel import load as load_panel
from feat import load_feat
import signals

COST = 0.0077; HOLD = 120; CS = (0.5, 2 / 3, 0.75); FAM = ('V', 'T', 'V∨T', 'V∧T')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); VALID = np.asarray(F['VALID'])
    AMT = np.asarray(P['AMT'], np.float64)
    Cf = pd.DataFrame(CA).ffill()
    Rr = (Cf / Cf.shift(1) - 1).where(pd.DataFrame(VALID))
    V20 = Rr.rolling(20, min_periods=15).std().to_numpy()
    dT = Tr - np.vstack([np.full((20, N), np.nan), Tr[:-20]])
    pct_cache = {}
    def pct(d):
        if d not in pct_cache:
            u = U[d]
            pv = pd.Series(np.where(u, V20[d], np.nan)).rank(pct=True).to_numpy()
            pt = pd.Series(np.where(u, dT[d], np.nan)).rank(pct=True).to_numpy()
            pct_cache[d] = (pv, pt)
        return pct_cache[d]
    D = M182.build(parity=1); D = D[D.q >= 2].reset_index(drop=True)
    below = M185.taiex_below(dates)
    cache = {}; rows = []
    for r in D.itertuples():
        S = cache.get(r.j) or cache.setdefault(r.j, M185.Stock(r.j, CA, OA, HA, LA, AMT, VALID))
        v = S.v; e = np.searchsorted(v, r.t + 1)
        if e >= len(v) or v[e] > r.t + 5: continue
        end = np.searchsorted(v, r.t + HOLD, side='right') - 1
        if end <= e + 1: continue
        ent = S.o[e]; c = S.c
        st = next((i for i in range(e, end + 1) if c[i] < ent * 0.85), None)
        x0 = st if st is not None else end
        p0 = (S.o[st + 1] if st + 1 <= end else c[st]) if st is not None else c[end]
        m0 = p0 / ent - 1 - COST
        brk = next((i for i in range(max(e, 1), end + 1) if below[v[i]] and not below[v[i - 1]]), None)
        row = dict(date=r.date, M0=m0, brk=brk is not None and brk < x0)
        if row['brk']:
            pe = (S.o[brk + 1] if brk + 1 <= end else c[brk]) / ent - 1 - COST
            pv, pt = pct(v[brk]); a, b = pv[r.j], pt[r.j]
            row.update(pv=a, pt=b)
            for cc in CS:
                hv = np.isfinite(a) and a >= cc; ht = np.isfinite(b) and b <= 1 - cc
                for fam, flag in (('V', hv), ('T', ht), ('V∨T', hv or ht), ('V∧T', hv and ht)):
                    row[f'{fam} c={cc:.2f}'] = pe if flag else m0
                    row[f'_x {fam} c={cc:.2f}'] = flag
        else:
            for cc in CS:
                for fam in FAM: row[f'{fam} c={cc:.2f}'] = m0; row[f'_x {fam} c={cc:.2f}'] = False
        rows.append(row)
    R = pd.DataFrame(rows)
    VS = ['M0'] + [f'{f} c={c:.2f}' for f in FAM for c in CS]
    L = ['# M-188 結果：大盤跌破多空點 × 個股 V20／ΔT 分別處理（R6 頭部剔除 Q1；奇數代碼；F 計數 69）\n',
         'V＝V20 百分位 ≥ c 出場；T＝ΔT 百分位 ≤ 1−c 出場；c＝0.50／0.67（主）／0.75。被出場者「續抱」＝其 M0 結果。\n']
    res = {}
    for era, lo, hi in M186.ERAS3:
        E = R[(R.date >= lo) & (R.date <= hi)]; B = E[E.brk]
        L += [f'## {era}（{len(E):,} 筆；遇跌破 {E.brk.mean():.0%}）\n',
              '| 版本 | 被出場比例（遇跌破者） | 平均 | 中位數 | 勝率 | P5 | 被出場者續抱：中位／平均 | 被保留者續抱：中位／平均 |', '|---|---|---|---|---|---|---|---|']
        for nm in VS:
            x = E[nm]; res[(nm, era)] = dict(med=x.median(), win=(x > 0).mean(), p5=x.quantile(.05))
            if nm == 'M0':
                L.append(f"| M0 不動 | | {x.mean():+.2%} | {x.median():+.2%} | {(x > 0).mean():.1%} | {x.quantile(.05):+.1%} | | |"); continue
            fx = B[f'_x {nm}'].astype(bool); ox, kp = B[fx].M0, B[~fx].M0
            L.append(f"| {nm} | {fx.mean():.0%} | {x.mean():+.2%} | {x.median():+.2%} | {(x > 0).mean():.1%} | {x.quantile(.05):+.1%} | {ox.median():+.1%}／{ox.mean():+.1%} | {kp.median():+.1%}／{kp.mean():+.1%} |")
        L.append('')
    vt = []
    for fam in FAM:
        ok = {c: all(res[(f'{fam} c={c:.2f}', e)]['med'] > res[('M0', e)]['med'] and res[(f'{fam} c={c:.2f}', e)]['win'] > res[('M0', e)]['win']
                     and res[(f'{fam} c={c:.2f}', e)]['p5'] >= res[('M0', e)]['p5'] - 0.005 for e, _, _ in M186.ERAS3) for c in CS}
        nom = ok[2 / 3] and sum(ok.values()) >= 2
        vt.append(f"- {fam}：" + '、'.join(f"c={c:.2f} {'✓' if ok[c] else '✗'}" for c in CS) + f" → {'提名' if nom else '不提名'}")
    head = ['## 判準（對 M0；三段都要：中位數↑、勝率↑、P5 不深於 −0.5pt；3 個 c 中 ≥ 2 且含 0.67）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline='；'.join(x[2:] for x in vt), n_variants=0)
