# M-204：分辨盤整、崩跌期與趨勢期的狀態指標。定義見 registry_M.md M-204。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
import sim, evalu as E
from poolret import pool_returns

NPATH = 100; END = '2024-11-13'; ERAS = ('2005–15', '2016–24/11')


def er(s, w):
    return (s - s.shift(w)).abs() / s.diff().abs().rolling(w).sum()


def main():
    P = load_panel(); dates = P['dates']; end = int(dates.searchsorted(pd.Timestamp(END), side='right'))
    out_r = {}
    for nm, xo in (('B0', None), ('F20', {'xfix': 20})):
        kw = dict(end=end)
        if xo: kw['xopt'] = xo
        R = sim.run_paths('fix', NPATH, sig='fix', x_kw=kw, v_kw=dict(end=end))
        out_r[nm] = E.path_mean([R['X'][i][0].loc[:END] for i in range(NPATH)])
    d = (out_r['F20'] - out_r['B0']).loc['2005-01-01':END]
    ld = np.log1p(out_r['F20']).cumsum() - np.log1p(out_r['B0']).cumsum()
    px = P['index']['TAIEX_P'].reindex(dates).ffill()
    df, _ = pool_returns(); su = np.log1p(df.S).cumsum() - np.log1p(df.U).cumsum()
    U250 = np.exp(np.log1p(df.U).cumsum().diff(250)) - 1
    IND = pd.DataFrame({'指數效率比ER60': er(px, 60), '指數60日報酬': px.pct_change(60), '指數60日波動': px.pct_change().rolling(60).std() * np.sqrt(252),
                        '指數距200MA': px / px.rolling(200).mean() - 1, 'U過去250日報酬': U250.reindex(dates), '強勢池相對U效率比60': er(su.reindex(dates), 60)})
    idx = ld.index; rows = []
    for i in range(260, len(idx) - 60, 20):
        t = idx[i]
        if t < pd.Timestamp('2005-06-01'): continue
        y = ld.iloc[i + 60] - ld.iloc[i]
        rows.append(dict(t=t, era=ERAS[0] if t <= pd.Timestamp('2015-12-31') else ERAS[1], y=y, **IND.loc[t].to_dict()))
    D = pd.DataFrame(rows).dropna()
    L = ['# M-204 結果：盤整、崩跌期與趨勢期的狀態指標（結果 = 未來 60 日「X1 固定 20 日 − X1 基準」累積對數報酬差；每 20 日取樣；正值代表快出場較好）\n',
         f"全體平均：{D[D.era == ERAS[0]].y.mean():+.2%}（{(D.era == ERAS[0]).sum()}）｜{D[D.era == ERAS[1]].y.mean():+.2%}（{(D.era == ERAS[1]).sum()}）\n",
         '| 指標 | 2005–15 最低 20%｜最高 20% | 2016–24/11 最低 20%｜最高 20% | 判定 |', '|---|---|---|---|']
    noms = []
    for k in IND.columns:
        lo, hi = D[k].quantile(0.2), D[k].quantile(0.8); cells, sg, ok = [], [], True
        for e in ERAS:
            a, b = D[(D.era == e) & (D[k] <= lo)].y, D[(D.era == e) & (D[k] >= hi)].y
            cells.append(f"{a.mean():+.2%}（{len(a)}）｜{b.mean():+.2%}（{len(b)}）")
            ok &= len(a) >= 15 and len(b) >= 15 and np.sign(a.mean()) != np.sign(b.mean()) and abs(a.mean() - b.mean()) >= 0.02
            sg.append((np.sign(a.mean()), np.sign(b.mean())))
        ok &= sg[0] == sg[1]
        if ok: noms.append(k)
        L.append(f"| {k}（≤ {lo:.3g}／≥ {hi:.3g}） | {cells[0]} | {cells[1]} | {'**提名**' if ok else ''} |")
    L += ['', '## 提名\n'] + ([f'- {n}' for n in noms] or ['- 無'])
    return dict(summary='\n'.join(L), headline=('提名：' + '、'.join(noms)) if noms else '無提名', n_variants=0)
