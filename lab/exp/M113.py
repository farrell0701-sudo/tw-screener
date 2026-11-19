# M-113：均線脈衝廣度（描述性）。定義見 registry_M.md M-113。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M107 import NAMES
from M106 import worst_dd

NP = 100; ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def states(F):
    Cfd = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); N = Cfd.shape[1]
    m120 = Cfd.rolling(120, min_periods=108).mean(); m200 = Cfd.rolling(200, min_periods=180).mean()
    above = (Cfd > m200).to_numpy() & m200.notna().to_numpy()
    g = (m120 / m200 - 1).to_numpy(); del m120, m200
    ag = np.abs(g); dag = ag - np.vstack([np.full((20, N), np.nan), ag[:-20]])
    st = np.full(g.shape, -1, np.int8); f = np.isfinite(dag) & np.isfinite(g); c = 0.01
    for base, side in ((0, g >= 0), (3, g < 0)):
        m = f & side; st[m & (dag < -c)] = base; st[m & (np.abs(dag) <= c)] = base + 1; st[m & (dag > c)] = base + 2
    return st, above


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    st, above = states(F)
    ok = st >= 0
    br = {}
    for nm, den in (('A', U & ok), ('B', pool & ok)):
        n = den.sum(1).astype(float); n[n == 0] = np.nan
        for k, s in enumerate(NAMES): br[f'{nm}{s}'] = pd.Series(((st == k) & den).sum(1) / n, index=dates)
    na = (U & ok).sum(1).astype(float); na[na == 0] = np.nan
    br['A站上200'] = pd.Series((above & U & ok).sum(1) / na, index=dates)
    br['A淨築底'] = br['A下收斂'] - br['A下發散']
    B = pd.DataFrame(br)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).astype(float)
    gi = ix.rolling(120).mean() / ix.rolling(200).mean() - 1
    R = sim.run_paths('fix', NP, sig='fix'); pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1).reindex(dates).fillna(0)
    rC = pm([sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NP)]); rX = pm([R['X'][i][0] for i in range(NP)]); rV = pm([R['V'][i][0] for i in range(NP)])
    fwd = lambda r: np.expm1(np.log1p(r)[::-1].rolling(20).sum()[::-1].shift(-1))
    FW = pd.DataFrame(dict(合併=fwd(rC), X1=fwd(rX), V=fwd(rV), 指數=fwd(ix.pct_change().fillna(0))))
    show = ['A上收斂', 'A下收斂', 'A淨築底', 'A站上200', 'B上收斂', 'B下收斂']
    L = ['# M-113 結果：均線脈衝廣度（描述性、不能提名）\n', 'A＝可交易池、B＝T_ld 強勢池；後 20 日＝t+1 起；新基準 100 路徑平均。\n']
    for era, lo, hi in ERAS:
        sel = (dates >= lo) & (dates <= hi); Bb = B[sel]; f = FW[sel]
        L += [f'## {era}\n', '### 1. 與大盤狀態的相關\n', '| 廣度 | 平均 | 與「大盤<200MA」相關 | 與大盤 g 相關 |', '|---|---|---|---|']
        for c in B.columns:
            L.append(f"| {c} | {Bb[c].mean():.1%} | {Bb[c].corr(below[sel]):+.2f} | {Bb[c].corr(gi[sel]):+.2f} |")
        L += ['\n### 2. 依廣度分五組的後 20 日報酬（合併｜X1｜V4.1｜指數）\n', '| 廣度 | Q1（低） | Q2 | Q3 | Q4 | Q5（高） | Q5−Q1 合併 |', '|---|---|---|---|---|---|---|']
        for c in show:
            q = pd.qcut(Bb[c].rank(method='first'), 5, labels=False); cells = []
            for k in range(5):
                x = f[q.values == k]; cells.append(f"{x.合併.mean():+.1%}｜{x.X1.mean():+.1%}｜{x.V.mean():+.1%}｜{x.指數.mean():+.1%}")
            d = f[q.values == 4].合併.mean() - f[q.values == 0].合併.mean()
            L.append(f"| {c} | " + ' | '.join(cells) + f" | {d:+.2%} |")
        # 4. 背離
        up = (below[sel] == 0); dv = (B['A站上200'] - B['A站上200'].shift(60))[sel] <= -0.10
        a, b = f[up & dv], f[up & ~dv]
        L += [f"\n### 4. 背離（大盤 ≥ 200MA 且站上 200 日線比例 60 日內降 ≥ 10pt）\n",
              f"背離日 {len(a)}（占大盤 ≥ 200MA 日 {len(a) / max(up.sum(), 1):.0%}）：後 20 日合併 {a.合併.mean():+.2%}、X1 {a.X1.mean():+.2%}、V4.1 {a.V.mean():+.2%}、指數 {a.指數.mean():+.2%}；"
              f"其他日：合併 {b.合併.mean():+.2%}、X1 {b.X1.mean():+.2%}、V4.1 {b.V.mean():+.2%}、指數 {b.指數.mean():+.2%}\n"]
    # 3. 回撤前
    eps = worst_dd(rC.loc['2005':], 5)
    L += ['## 3. 合併版前 5 大回撤前的強勢池上收斂廣度（百分位＝在所屬期間的位置）\n', '| 回撤 | 高點 | 前 60 日 | 前 20 日 | 高點日 | 低點日 |', '|---|---|---|---|---|---|']
    s = B['B上收斂']
    for pk, tr, dd in eps:
        lo, hi = ('2005', '2015') if pk.year <= 2015 else ('2016', '2026'); ref = s.loc[lo:hi].dropna()
        pct = lambda d: f"{s.loc[d]:.0%}（P{(ref < s.loc[d]).mean() * 100:.0f}）"
        t = dates.get_loc(pk)
        L.append(f"| {dd:.1%} | {pk.date()} | {pct(dates[t - 60])} | {pct(dates[t - 20])} | {pct(pk)} | {pct(tr)} |")
    return dict(summary='\n'.join(L), headline='廣度描述完成', n_variants=0)
