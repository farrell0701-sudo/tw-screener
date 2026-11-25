# M-200：X1 與合併版的系統性回撤與期貨避險疊加。定義見 registry_M.md M-200。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
import sim, evalu as E

END = '2024-11-13'; NPATH = 100
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', END), ('16–24/11', '2016-01-01', END))


def rbeta(y, x, w=250):
    return (y.rolling(w, min_periods=120).cov(x) / x.rolling(w, min_periods=120).var()).shift(1)


def dd_table(r, rix, b, k=5):
    e = (1 + r).cumprod(); dd = e / e.cummax() - 1; rows = []; used = pd.Series(False, index=r.index)
    for _ in range(k):
        x = dd.where(~used)
        if x.min() >= -0.03 or x.isna().all(): break
        tr = x.idxmin(); pk = e.loc[:tr].idxmax()
        rec = e.loc[tr:][e.loc[tr:] >= e.loc[pk]].index; end = rec[0] if len(rec) else e.index[-1]
        used.loc[pk:end] = True
        seg = slice(pk, tr); tot = e.loc[tr] / e.loc[pk] - 1
        mkt = (1 + (b.loc[seg] * rix.loc[seg]).fillna(0)).prod() - 1
        rows.append((pk, tr, tot, mkt, (1 + rix.loc[seg]).prod() - 1))
    return rows


def main():
    P = load_panel(); dates = P['dates']
    end = int(dates.searchsorted(pd.Timestamp(END), side='right'))
    R = sim.run_paths('fix', NPATH, sig='fix', x_kw=dict(end=end), v_kw=dict(end=end))
    X = E.path_mean([R['X'][i][0].loc[:END] for i in range(NPATH)])
    B = E.path_mean([sim.mix(R['X'][i][0].loc[:END], R['V'][i][0].loc[:END]) for i in range(NPATH)])
    ix = P['index'][['TAIEX_P', 'TAIEX_TR']].reindex(dates).ffill()
    rix = ix['TAIEX_TR'].pct_change().reindex(X.index).fillna(0)
    px = ix['TAIEX_P'].reindex(X.index)
    st = {'H2 低於200MA': px < px.rolling(200).mean(), 'H3 低於60MA': px < px.rolling(60).mean(), 'H4 回落≥10%': px <= px.rolling(250, min_periods=60).max() * 0.9}
    out = ['# M-200 結果：X1 與合併版的系統性回撤與期貨避險疊加（100 路徑平均日報酬；到 2024-11-13；避險 = 放空 滾動β × TAIEX_TR，訊號用前一日收盤，忽略期貨成本）\n']
    for nm, r in (('X1', X), ('合併版', B)):
        b = rbeta(r, rix)
        out += [f'## {nm}\n', f"滾動 beta：中位數 {b.median():.2f}；2005–15 {b.loc['2005':'2015'].median():.2f}；2016–24/11 {b.loc['2016':].median():.2f}\n",
                '### 最深 5 次回撤（峰 → 谷）\n', '| 期間 | 回撤 | 市場貢獻（β×指數） | 剩餘 | 同期指數 |', '|---|---|---|---|---|']
        for pk, tr, tot, mkt, ixr in dd_table(r, rix, b):
            out.append(f"| {pk:%Y-%m-%d} → {tr:%Y-%m-%d} | {tot:.1%} | {mkt:+.1%} | {(1 + tot) / (1 + mkt) - 1:+.1%} | {ixr:+.1%} |")
        V = {'H0 不避險': r, 'H1 永遠避險': r - b.fillna(0) * rix}
        for k, s in st.items(): V[k] = r - (b.fillna(0) * rix).where(s.shift(1).fillna(False).astype(bool), 0)
        out += ['', '### 疊加版本（年化／最大回撤／Calmar）\n', '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' | 避險天數比例 |', '|---|---|---|---|---|---|']
        for k, s in V.items():
            frac = {'H0 不避險': 0, 'H1 永遠避險': 1}.get(k, None)
            if frac is None: frac = st[k].loc['2005':].mean()
            out.append(f'| {k} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*E.met(s.loc[lo:hi])[:3]) for _, lo, hi in SEGS) + f' | {frac:.0%} |')
        out.append('')
    return dict(summary='\n'.join(out), headline='系統性回撤拆解與避險疊加描述', n_variants=0)
