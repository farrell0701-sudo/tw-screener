# M-109：大盤週 RSI(14) 跌破 30 → 60 個交易日內解除 X1 的 200MA 閘門（訊號分析，探索＝奇數、確認＝偶數，2005–15）。判準見 registry_M.md M-109。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M106 import worst_dd

START, END = pd.Timestamp('2005-01-01'), pd.Timestamp('2015-12-31')
TH, WIN, NP = 30, 60, 100


def wrsi(ix, n=14):
    d = ix.diff(); u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / dn)


def main():
    rng = np.random.default_rng(20261119)
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T = len(dates)
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    wk = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max()      # 每週最後交易日
    w = ix.loc[wk.values]; r = wrsi(w); trig = r.index[(r < TH) & (r.shift(1) >= TH)]
    unlock = np.zeros(T, bool)
    for d in trig:
        s = dates.get_loc(d); unlock[s:s + WIN] = True
    below = (ix < ix.rolling(200).mean()).to_numpy()
    B = sim.run_paths('fix', NP, sig='fix'); U = sim.run_paths('fix', NP, sig='fix', x_kw={'unlock': unlock})
    odd = {k: int(str(c)[-1]) % 2 == 1 for k, c in zip(P['meta']['security_key'], P['meta']['stock_id'].astype(str))}

    def dedup(R):
        seen = {}
        for i in range(NP):
            for (code, d0, d1, wt, ret, hold, why, st, lk) in R['X'][i][1]:
                seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret))
        X = pd.DataFrame(seen.values()); t0 = dates.get_indexer(X.d0)
        X['extra'] = below[t0 - 1]; X['odd'] = X.code.map(odd); X['m'] = X.d0.astype(str).str[:7]
        return X
    XB, XU = dedup(B), dedup(U)
    rb = pd.concat([B['X'][i][0] for i in range(NP)], axis=1).mean(axis=1)
    ru = pd.concat([U['X'][i][0] for i in range(NP)], axis=1).mean(axis=1)
    eps = worst_dd(rb.loc[START:END])

    def judge(par, lo, hi):
        sub = lambda X: X[(X.odd == par) & (X.d0 >= lo) & (X.d1 <= hi)]
        base = sub(XB); E = sub(XU); E = E[E.extra]
        if len(E) == 0: return [False] * 4, '額外交易 0 筆', E
        q10, q90 = base.ret.quantile([0.1, 0.9])
        mean = E.ret.mean()
        # 月份群集 bootstrap：P(平均 > 0)
        g = {m: x.ret.values for m, x in E.groupby('m')}; ms = list(g); cnt = 0
        for _ in range(5000):
            pick = rng.choice(len(ms), len(ms)); v = np.concatenate([g[ms[k]] for k in pick]); cnt += v.mean() > 0
        p = cnt / 5000
        left, right = (E.ret <= q10).mean(), (E.ret >= q90).mean()
        inn = E[[any(x.d0 <= tr and x.d1 >= pk for pk, tr, _ in eps) for x in E.itertuples()]]
        ddm = inn.ret.mean() if len(inn) else 0.0
        ok = [mean > 0 and p >= 0.90, left <= 0.15 and right >= 0.10, ddm >= 0, len(E) >= 30]
        rows = ['| 判準 | 數值 | 門檻 | 結果 |', '|---|---|---|---|',
                f"| (a) 平均報酬 | {mean:+.2%}；月份群集 P {p:.3f} | > 0 且 P ≥ 0.90 | {'✓' if ok[0] else '✗'} |",
                f"| (b) 左尾率／右尾率 | {left:.0%}／{right:.0%} | ≤ 15%／≥ 10% | {'✓' if ok[1] else '✗'} |",
                f"| (c) 三次最大回撤期間內平均 | {ddm:+.2%}（{len(inn)} 筆） | ≥ 0 | {'✓' if ok[2] else '✗'} |",
                f"| (d) 筆數 | {len(E)} | ≥ 30 | {'✓' if ok[3] else '✗'} |"]
        return ok, '\n'.join(rows), E

    L = ['# M-109 結果：大盤週 RSI(14) 跌破 30 → 60 日內解除 X1 的 200MA 閘門\n',
         '觸發日：' + '、'.join(str(d.date()) for d in trig) + '\n',
         'X1 三次最大回撤（基準，2005–15）：' + '；'.join(f'{a.date()}→{b.date()} {d:.1%}' for a, b, d in eps) + '\n']
    ok1, t1, E1 = judge(True, START, END)
    L += ['## 探索集（奇數代碼 2005–15）\n', t1, f"\n**{'提名，進確認集' if all(ok1) else '不提名，結案（不跑確認集判定）'}**\n"]
    ok2 = None
    if all(ok1):
        ok2, t2, _ = judge(False, START, END)
        L += ['## 確認集（偶數代碼 2005–15，只測一次）\n', t2, f"\n**{'通過，可登記 L 系列' if all(ok2) else '不通過，結案'}**\n"]
    # 另報：每次觸發
    L += ['## 另報：各次觸發的額外交易（全部代碼、全期，以進場日歸屬最近一次觸發）\n', '| 觸發日 | 額外交易 | 平均報酬 | 勝率 |', '|---|---|---|---|']
    EA = XU[XU.extra].copy(); tg = pd.DatetimeIndex(trig)
    EA['trig'] = [tg[tg <= d][-1] if (tg <= d).any() else pd.NaT for d in EA.d0]
    for d, x in EA.groupby('trig'):
        L.append(f"| {d.date()} | {len(x)} | {x.ret.mean():+.2%} | {(x.ret > 0).mean():.0%} |")
    def st(r, lo, hi):
        r = r.loc[lo:hi]; eq = (1 + r).cumprod(); yrs = len(r) / 245
        return f"{eq.iloc[-1] ** (1 / yrs) - 1:+.1%} / {(eq / eq.cummax() - 1).min():.1%}"
    L += ['\n## 另報：X1 子帳戶（100 路徑平均；年化 / 回撤）\n', '| 期間 | 基準 | 解除 |', '|---|---|---|']
    for nm, lo, hi in (('2005–15', '2005', '2015'), ('2016–26（僅參考）', '2016', '2026')):
        L.append(f"| {nm} | {st(rb, lo, hi)} | {st(ru, lo, hi)} |")
    head = f"探索集{'提名' if all(ok1) else '不提名'}" + ('' if ok2 is None else f"；確認集{'通過' if all(ok2) else '不通過'}")
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
