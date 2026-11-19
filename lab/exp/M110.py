# M-110：週 RSI 回升確認後 60 日內解除 X1 的 200MA 閘門（描述性）。定義見 registry_M.md M-110。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals, sim

WIN, NP = 60, 100
VARS = {'A：RSI14 回到 30': (14, 30, 30), 'B：RSI14 回到 40': (14, 30, 40), 'C：RSI6 回到 20': (6, 20, 20), 'D：拿掉閘門（參考）': None}


def wrsi(x, n):
    d = x.diff(); u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / dn)


def triggers(r, arm, fire):
    out, armed = [], False
    for d, v in r.items():
        if v < arm: armed = True
        elif armed and v >= fire: out.append(d); armed = False
    return out


def stats(r, lo, hi):
    r = r.loc[lo:hi]; eq = (1 + r).cumprod(); a = eq.iloc[-1] ** (245 / len(r)) - 1; dd = (eq / eq.cummax() - 1).min()
    return a, dd, a / -dd


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T = len(dates)
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    wk = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max(); w = ix.loc[wk.values]
    pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    B = sim.run_paths('fix', NP, sig='fix'); rV = [B['V'][i][0] for i in range(NP)]
    res = {'基準': (pm([B['X'][i][0] for i in range(NP)]), pm([sim.mix(B['X'][i][0], rV[i]) for i in range(NP)]), None, [])}
    for nm, spec in VARS.items():
        if spec is None: unlock = np.ones(T, bool); trig = []
        else:
            n, arm, fire = spec; trig = [d for d in triggers(wrsi(w, n), arm, fire) if below[dates.get_loc(d)]]
            unlock = np.zeros(T, bool)
            for d in trig: s = dates.get_loc(d); unlock[s:s + WIN] = True
        U = sim.run_paths('fix', NP, sig='fix', x_kw={'unlock': unlock})
        seen = {}
        for i in range(NP):
            for (code, d0, d1, wt, ret, *_r) in U['X'][i][1]: seen.setdefault((code, d0), dict(d0=d0, ret=ret))
        X = pd.DataFrame(seen.values()); X = X[below[dates.get_indexer(X.d0) - 1]]
        res[nm] = (pm([U['X'][i][0] for i in range(NP)]), pm([sim.mix(U['X'][i][0], rV[i]) for i in range(NP)]), X, trig)
    L = ['# M-110 結果：週 RSI 回升確認後 60 日內解除 X1 的 200MA 閘門（描述性、不能提名）\n',
         '## X1 子帳戶與合併版（100 路徑平均日報酬；年化 / 回撤 / Calmar）\n',
         '| 版本 | X1 2005–15 | X1 2016–26 | 合併 2005–15 | 合併 2016–26 | 額外交易（05–15／16–26） |', '|---|---|---|---|---|---|']
    f = lambda t: f"{t[0]:+.1%} / {t[1]:.1%} / {t[2]:.2f}"
    for nm, (rx, rc, X, trig) in res.items():
        ne = '—' if X is None else f"{(X.d0 < '2016').sum()}／{(X.d0 >= '2016').sum()}"
        L.append(f"| {nm} | {f(stats(rx, '2005', '2015'))} | {f(stats(rx, '2016', '2026'))} | {f(stats(rc, '2005', '2015'))} | {f(stats(rc, '2016', '2026'))} | {ne} |")
    for nm, (rx, rc, X, trig) in res.items():
        if not trig: continue
        L += [f'\n## {nm}：各次觸發\n', '| 觸發日 | 額外交易 | 平均報酬 | 勝率 |', '|---|---|---|---|']
        tg = pd.DatetimeIndex(trig); X = X.copy(); X['tg'] = [tg[tg <= d][-1] if (tg <= d).any() else pd.NaT for d in X.d0]
        for d in tg:
            x = X[X.tg == d]
            L.append(f"| {d.date()} | {len(x)} | {x.ret.mean():+.2%} | {(x.ret > 0).mean():.0%} |" if len(x) else f"| {d.date()} | 0 | — | — |")
    return dict(summary='\n'.join(L), headline='回升確認描述完成', n_variants=0)
