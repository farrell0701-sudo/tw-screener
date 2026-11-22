# M-174：大盤多空點（2 個月月份階梯）翻轉拆解（描述性）。定義見 registry_M.md M-174。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel


def series(eid, name):
    z = np.load(LAB / 'results' / eid / 'series.npz', allow_pickle=True)
    idx = pd.to_datetime(z['dates'].astype('datetime64[D]')); nm = list(z['names'])
    return pd.Series(z['values'][:, nm.index(name)], index=idx)


def main():
    P = load_panel(); dates = P['dates']
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    mon = dates.to_period('M'); cm = ix.groupby(mon).agg(['max', 'min'])
    x = ((cm['max'].rolling(2).max().shift(1) + cm['min'].rolling(2).min().shift(1)) / 2)
    xd = pd.Series(x.reindex(mon).to_numpy(), index=dates); xprev = pd.Series(x.shift(1).reindex(mon).to_numpy(), index=dates)
    bear = (ix < xd) & xd.notna()
    N1 = series('L134', 'S_N1'); K0 = series('L132', 'S_E21')
    first = pd.Series(mon, index=dates) != pd.Series(mon, index=dates).shift(1)
    rows = []
    b = bear.to_numpy(); n = len(dates)
    for t in range(1, n):
        if b[t] == b[t - 1] or not np.isfinite(xd.iloc[t]) or dates[t] < pd.Timestamp('2005-01-01'): continue
        cause = '穿越'
        if first.iloc[t] and np.isfinite(xprev.iloc[t]) and (ix.iloc[t] < xprev.iloc[t]) == b[t - 1]: cause = '換線'
        nxt = np.nonzero(b[t + 1:] != b[t])[0]; dur = int(nxt[0]) + 1 if len(nxt) else n - t
        r = lambda k: float(ix.iloc[min(t + k, n - 1)] / ix.iloc[t] - 1)
        seg = slice(dates[t + 1] if t + 1 < n else dates[t], dates[min(t + dur, n - 1)])
        rows.append(dict(date=dates[t], dir='翻空' if b[t] else '翻多', cause=cause, r5=r(5), r20=r(20), false=dur <= 20, dur=dur,
                         n1=float((1 + N1.loc[seg]).prod() - 1), k0=float((1 + K0.loc[seg]).prod() - 1), ixr=float(ix.loc[seg.stop] / ix.iloc[t] - 1)))
    D = pd.DataFrame(rows)
    D['era'] = np.where(D.date < '2016-01-01', '2005–15', np.where(D.date < '2020-01-01', '2016–19', '2020–26'))
    yrs = {'2005–15': 11, '2016–19': 4, '2020–26': (dates[-1] - pd.Timestamp('2020-01-01')).days / 365.25}
    L = ['# M-174 結果：大盤多空點（2 個月月份階梯）翻轉拆解\n', '## 各期間彙整\n',
         '| 期間 | 翻轉／年 | 換線占比 | 假訊號率（20 日內翻回） | 翻空後 20 日指數 | 翻多後 20 日指數 | 翻空期間 指數 | 翻空期間 N1 − K0 | 換線翻空 假訊號率 | 穿越翻空 假訊號率 |', '|---|---|---|---|---|---|---|---|---|---|']
    for era, g in D.groupby('era'):
        dn, up = g[g.dir == '翻空'], g[g.dir == '翻多']
        fc = lambda h: f'{h.false.mean():.0%}（{len(h)}）' if len(h) else '—'
        L.append(f"| {era} | {len(g) / yrs[era]:.1f} | {(g.cause == '換線').mean():.0%} | {g.false.mean():.0%} | {dn.r20.mean():+.2%} | {up.r20.mean():+.2%} | {dn.ixr.mean():+.2%} | {(dn.n1 - dn.k0).mean():+.2%} | "
                 f"{fc(dn[dn.cause == '換線'])} | {fc(dn[dn.cause == '穿越'])} |")
    L += ['', '## 翻空期間的總效果（依期間，N1 與 K0 在翻空期間的累計報酬）\n', '| 期間 | 翻空次數 | 翻空期間 N1 累計 | 同期間 K0 累計 | 翻多期間 N1 累計 | 翻多期間 K0 累計 |', '|---|---|---|---|---|---|']
    for era, g in D.groupby('era'):
        dn, up = g[g.dir == '翻空'], g[g.dir == '翻多']
        L.append(f"| {era} | {len(dn)} | {(1 + dn.n1).prod() - 1:+.1%} | {(1 + dn.k0).prod() - 1:+.1%} | {(1 + up.n1).prod() - 1:+.1%} | {(1 + up.k0).prod() - 1:+.1%} |")
    L += ['', '## 2016–19 逐筆\n', '| 日期 | 方向 | 原因 | 持續天數 | 後 5 日指數 | 後 20 日指數 | 狀態期間 指數 | 狀態期間 N1 | 狀態期間 K0 |', '|---|---|---|---|---|---|---|---|---|']
    for _, r in D[D.era == '2016–19'].iterrows():
        L.append(f"| {r.date:%Y-%m-%d} | {r.dir} | {r.cause} | {r.dur} | {r.r5:+.1%} | {r.r20:+.1%} | {r.ixr:+.1%} | {r.n1:+.1%} | {r.k0:+.1%} |")
    return dict(summary='\n'.join(L), headline='多空點翻轉拆解完成', n_variants=0)
