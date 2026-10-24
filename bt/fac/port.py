"""最簡單的排序組合回測
分數 = T_ld 與 S_ovh 的當日橫斷面百分位平均（等權、無參數）
每 20 個交易日換股：t 收盤算分數 → t+1 開盤買進前 N 檔、等權；開盤漲停買不到就往下遞補
成本：手續費 0.0225%/邊、證交稅 0.3%（賣）、滑價 0.1%/邊
20 個不同起始日各跑一次，看路徑運氣"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare
from factors import build

FEE, TAX, SLIP = 0.000225, 0.003, 0.001
REB = 20

P = load_panel(); E = prepare(P); C, O = P['C'], P['O']; U = E['univ']
Cf = C.ffill()
D = pd.read_pickle('/home/claude/bt/struct/daily.pkl')
D = {k: v.reindex(index=C.index, columns=C.columns) for k, v in D.items()}
F = build(P, D)
pct = lambda k: (F[k][0] * F[k][1]).where(U).rank(axis=1, pct=True)
SC = {'合成': (pct('T_ld') + pct('S_ovh')) / 2, 'T_ld': pct('T_ld'), 'S_ovh': pct('S_ovh')}
ind = P['industry'].reindex(C.columns).fillna('NA').values
dates = C.index; d0 = dates.get_indexer([pd.Timestamp('2016-02-01')], method='bfill')[0]
Ov, Cfv_ = O.values, Cf.values
buyable = (O.shift(-1) / C - 1 < 0.095) & O.shift(-1).notna()          # t+1 開盤有成交且非漲停
BUY = buyable.values


def backtest(S, N, off, cost=True):
    Sv = S.values if S is not None else U.values.astype(float)
    reb = list(range(d0 + off, len(dates) - 2, REB))
    eq = np.full(len(dates), np.nan); val = 1.0; held = set(); rows = []
    for j, t in enumerate(reb):
        e = t + 1                                                # 成交日
        x = reb[j + 1] + 1 if j + 1 < len(reb) else len(dates) - 1
        sc = np.where(BUY[t] & U.values[t], Sv[t], np.nan)
        ok = np.where(~np.isnan(sc))[0]
        pick = ok[np.argsort(-sc[ok])[:N]] if N else ok
        new = set(pick)
        if cost:
            sold = len(held - new) / max(1, len(held)) if held else 0
            bought = len(new - held) / len(new)
            val *= 1 - sold * (FEE + TAX + SLIP) - bought * (FEE + SLIP)
        px0 = Ov[e, pick]
        path = Cfv_[e:x + 1, pick] / px0                       # 逐日市值（停牌/下市以最後價計）
        path[-1] = np.where(np.isnan(Ov[x, pick]), Cfv_[x, pick], Ov[x, pick]) / px0   # 下次換股日開盤賣出
        curve = val * np.nanmean(path, axis=1)
        eq[e:x + 1] = curve; val = curve[-1]; held = new
        if N: rows.append(dict(date=dates[t], top_ind_share=pd.Series(ind[pick]).value_counts().iloc[0] / len(pick)))
    s = pd.Series(eq, index=dates).dropna()
    return s, pd.DataFrame(rows)


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    return dict(CAGR=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, MDD=(s / s.cummax() - 1).min(),
                Vol=r.std() * np.sqrt(252), Sharpe=r.mean() / r.std() * np.sqrt(252))


if __name__ == '__main__':
    out = []; curves = {}
    bench = {off: backtest(None, 0, off, cost=False)[0] for off in range(REB)}
    for name in ['合成', 'T_ld', 'S_ovh']:
        for N in (10, 20, 50):
            for off in range(REB):
                s, h = backtest(SC[name], N, off)
                b = bench[off].reindex(s.index)
                st = stats(s); bs = stats(b)
                rel = (s / s.iloc[0]) / (b / b.iloc[0])
                mo = np.log(rel).resample('ME').last().diff().dropna()
                yr = np.log(rel).resample('YE').last().diff()
                yr.iloc[0] = np.log(rel).resample('YE').last().iloc[0]
                pos = mo.clip(lower=0).sort_values(ascending=False)
                out.append(dict(score=name, N=N, off=off, **st, 超額CAGR=st['CAGR'] - bs['CAGR'], 基準CAGR=bs['CAGR'],
                                超額正月=(mo > 0).mean(), 前10pct月貢獻=pos.iloc[:len(mo) // 10].sum() / mo.sum() if mo.sum() > 0 else np.nan,
                                超額正年=(yr > 0).sum(), 年數=len(yr), 最差年超額=yr.min(),
                                前半超額=np.log(rel[:'2020'].iloc[-1]) / 5, 後半超額=(np.log(rel.iloc[-1]) - np.log(rel[:'2020'].iloc[-1])) / (len(rel['2021':]) / 252),
                                單一產業占比=h.top_ind_share.mean()))
                if off == 0: curves[(name, N)] = (s, b, yr)
            print(name, N, 'done', flush=True)
    R = pd.DataFrame(out); R.to_pickle('/home/claude/bt/fac/port_runs.pkl')
    pd.to_pickle(curves, '/home/claude/bt/fac/port_curves.pkl')
    g = R.groupby(['score', 'N'])
    summ = g.agg(CAGR中位=('CAGR', 'median'), CAGR最低=('CAGR', 'min'), CAGR最高=('CAGR', 'max'), MDD中位=('MDD', 'median'),
                 Sharpe=('Sharpe', 'median'), 超額CAGR=('超額CAGR', 'median'), 基準CAGR=('基準CAGR', 'median'),
                 超額正月=('超額正月', 'median'), 前10pct月貢獻=('前10pct月貢獻', 'median'), 超額正年=('超額正年', 'median'),
                 最差年超額=('最差年超額', 'median'), 前半超額=('前半超額', 'median'), 後半超額=('後半超額', 'median'),
                 單一產業占比=('單一產業占比', 'median'))
    pd.set_option('display.width', 250); print(summ.round(3).to_string())
    summ.to_csv('/home/claude/bt/fac/port_summary.csv', encoding='utf-8-sig')
