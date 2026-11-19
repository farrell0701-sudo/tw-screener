# M-144：X1 閘門換成週 EMA 下彎（訊號分析，探索集奇數代碼 2005–15）。定義見 registry_M.md M-144。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M106 import worst_dd

NP = 100; ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def weekly_gate(ix, dates, n):
    wend = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max()
    wc = ix.loc[wend.values]; e = wc.ewm(span=n, adjust=False).mean(); fl = (e < e.shift(1)); fl.index = wend.values
    return fl.reindex(dates, method='ffill').fillna(False).to_numpy(bool)


def boot(vals, months, rng, nb=5000):
    g = pd.Series(vals).groupby(np.asarray(months)).apply(lambda x: x.to_numpy()); ks = list(g.index); cnt = 0
    for _ in range(nb):
        pick = rng.choice(len(ks), len(ks)); cnt += np.concatenate([g.iloc[k] for k in pick]).mean() > 0
    return cnt / nb


def main():
    rng = np.random.default_rng(20261120)
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    odd = {k: int(str(c)[-1]) % 2 == 1 for k, c in zip(P['meta']['security_key'], P['meta']['stock_id'].astype(str))}
    def dedup(R):
        seen = {}
        for i in range(NP):
            for (code, d0, d1, w, ret, *_r) in R['X'][i][1]: seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret))
        X = pd.DataFrame(seen.values()); X['td'] = dates.get_indexer(X.d0) - 1; X['odd'] = X.code.map(odd); X['m'] = X.d0.astype(str).str[:7]
        return X
    B = sim.run_paths('fix', NP, sig='fix'); XB = dedup(B)
    rb = pd.concat([B['X'][i][0] for i in range(NP)], axis=1).mean(axis=1)
    eps = worst_dd(rb.loc['2005':'2015'])
    L = ['# M-144 結果：X1 閘門換成週 EMA 下彎（訊號分析）\n', 'X1 三次最大回撤（基準 2005–15）：' + '；'.join(f'{a.date()}→{b.date()} {d:.1%}' for a, b, d in eps) + '\n']
    verdict = None
    for n in (21, 16, 26):
        gw = weekly_gate(ix, dates, n)
        G = sim.run_paths('fix', NP, sig='fix', x_kw={'gate': gw}); XG = dedup(G)
        XG['extra'] = below[XG.td] & ~gw[XG.td]
        XB['removed'] = ~below[XB.td] & gw[XB.td]
        rg = pd.concat([G['X'][i][0] for i in range(NP)], axis=1).mean(axis=1)
        L.append(f'## 週 EMA{n}{"（主版本）" if n == 21 else "（高原）"}\n')
        for era, lo, hi in ERAS:
            for par, pn in ((True, '奇數（探索集）'), (None, '全部代碼')):
                if era == '2016–26' and par is True: continue
                f = lambda X: X[((X.odd == par) if par is not None else True) & (X.d0 >= lo) & (X.d1 <= hi)]
                base = f(XB); ext = f(XG); ext = ext[ext.extra]; rem = base[base.removed]
                q10, q90 = base.ret.quantile([0.1, 0.9])
                p = boot(ext.ret.values, ext.m.values, rng) if len(ext) > 5 else np.nan
                inn = ext[[any(x.d0 <= tr and x.d1 >= pk for pk, tr, _ in eps) for x in ext.itertuples()]] if era == '2005–15' else ext.iloc[:0]
                ddm = inn.ret.mean() if len(inn) else 0.0
                crit = [ext.ret.mean() > 0 and p >= 0.90, (ext.ret <= q10).mean() <= 0.15 and (ext.ret >= q90).mean() >= 0.10, ddm >= 0, len(ext) >= 30, rem.ret.mean() < base.ret.mean()]
                L += [f'### {era}｜{pn}\n', '| 項目 | 數值 | 判準 |', '|---|---|---|',
                      f"| (a) 多出來的交易平均（P） | {ext.ret.mean():+.2%}（{p:.3f}），{len(ext)} 筆 | {'✓' if crit[0] else '✗'} |",
                      f"| (b) 左尾率／右尾率 | {(ext.ret <= q10).mean():.0%}／{(ext.ret >= q90).mean():.0%} | {'✓' if crit[1] else '✗'} |",
                      f"| (c) 三次最大回撤期間內平均 | {ddm:+.2%}（{len(inn)} 筆） | {'✓' if crit[2] else '✗'} |" if era == '2005–15' else '| (c) | — | — |',
                      f"| (d) 筆數 ≥ 30 | {len(ext)} | {'✓' if crit[3] else '✗'} |",
                      f"| (e) 少掉的交易平均 vs 基準全部 | {rem.ret.mean():+.2%}（{len(rem)} 筆） vs {base.ret.mean():+.2%} | {'✓' if crit[4] else '✗'} |", '']
                if n == 21 and era == '2005–15' and par is True: verdict = all(crit)
        st = lambda r, lo, hi: (lambda x: f"{(1 + x).prod() ** (245 / len(x)) - 1:+.1%} / {((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min():.1%}")(r.loc[lo:hi])
        L += ['| X1 子帳戶（年化／回撤） | 基準 | 新閘門 |', '|---|---|---|', f"| 2005–15 | {st(rb, '2005', '2015')} | {st(rg, '2005', '2015')} |", f"| 2016–26 | {st(rb, '2016', '2026')} | {st(rg, '2016', '2026')} |", '']
    L.append(f"**主版本（EMA21、探索集 2005–15）：{'提名，送偶數代碼確認集' if verdict else '不提名，結案'}**")
    return dict(summary='\n'.join(L), headline=f"週 EMA 閘門{'提名' if verdict else '不提名'}", n_variants=0)
