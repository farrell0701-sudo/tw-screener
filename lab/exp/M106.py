# M-106：「120 日線在 200 日線下方時不進場」確認集（偶數代碼）只測一次。R1＝V4.1、R2＝X1。判準見 registry_M.md M-106。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals, sim

START, END = pd.Timestamp('2005-01-01'), pd.Timestamp('2015-12-31')
K = 60; NB = 5000; SEED = 20261119


def boot_p(dates_m, val, flag, rng):
    """月份群集 bootstrap：P(保留平均 − 標記平均 > 0)。"""
    df = pd.DataFrame(dict(m=dates_m, v=val, f=flag)); months = df.m.unique()
    groups = {m: g for m, g in df.groupby('m')}
    cnt = 0
    for _ in range(NB):
        pick = rng.choice(months, len(months), replace=True)
        d = pd.concat([groups[m] for m in pick])
        if d.f.any() and (~d.f).any():
            cnt += (d.v[~d.f].mean() - d.v[d.f].mean()) > 0
    return cnt / NB


def worst_dd(r, n=3):
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    eps = []; used = pd.Series(False, index=dd.index)
    for _ in range(n):
        c = dd[~used]
        if c.empty or c.min() >= 0: break
        tr = c.idxmin(); pk = eq.loc[:tr].idxmax(); eps.append((pk, tr, float(dd.loc[tr])))
        nxt = eq.loc[tr:]; rec = nxt[nxt >= eq.loc[pk]]
        used.loc[pk:(rec.index[0] if len(rec) else dd.index[-1])] = True
    return eps


def main():
    rng = np.random.default_rng(SEED)
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, signal = dbg['U'], dbg['signal']
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64)
    Cfd = pd.DataFrame(CA).ffill(); Cf = Cfd.to_numpy()
    m120 = Cfd.rolling(120, min_periods=108).mean(); m200 = Cfd.rolling(200, min_periods=180).mean()
    g = (m120 / m200 - 1).to_numpy(); del m120
    s120 = None
    s200 = (m200 / m200.shift(20) - 1).to_numpy(); del m200
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    sid = P['meta']['stock_id'].astype(str); even = (sid.str[-1].astype(int) % 2 == 0).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1
    kix = {k: j for j, k in enumerate(P['meta']['security_key'])}
    L = ['# M-106 結果：120 日線在 200 日線下方時不進場（確認集：偶數代碼 2005–15，只測一次）\n']

    Rp = sim.run_paths('fix', 100, sig='fix')
    def trades(kind):
        seen = {}
        for i in range(100):
            for (code, d0, d1, w, ret, hold, why, st, lk) in Rp[kind][i][1]:
                seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret))
        rows = []
        for v in seen.values():
            j = kix[v['code']]; t0 = int(dates.get_loc(v['d0'])); t1 = int(dates.get_loc(v['d1']))
            if not even[j] or t0 < t_lo or t1 > t_hi: continue
            s = t0 - 1
            rows.append(dict(j=j, t0=t0, t1=t1, ret=v['ret'], flag=bool(g[s, j] < 0) if np.isfinite(g[s, j]) else False,
                             nonup=bool(not (np.isfinite(g[s, j]) and s200[s, j] > 0)), m=str(v['d0'])[:7]))
        return pd.DataFrame(rows)
    def ddc(X, kind):
        r = pd.concat([Rp[kind][i][0] for i in range(100)], axis=1).mean(axis=1).loc[START:END]
        eps = worst_dd(r)
        def per(sub):
            tot, n = 0.0, 0
            for _, x in sub.iterrows():
                hit = False
                for pk, tr, _ in eps:
                    a, b = int(dates.get_loc(pk)), int(dates.get_loc(tr)); s_, e_ = max(x.t0, a), min(x.t1, b)
                    if s_ < e_: tot += Cf[e_, x.j] / Cf[s_, x.j] - 1; hit = True
                n += hit
            return tot / max(n, 1), n
        return eps, per(X[X.flag]), per(X[~X.flag])

    # ---------- R1：V4.1 ----------
    tt, jj = np.nonzero(signal)
    ok = (tt - K >= t_lo) & (tt + 20 <= t_hi) & even[jj]; tt, jj = tt[ok], jj[ok]
    r20 = CX[tt + 20, jj] - CX[tt, jj]
    fl = np.where(np.isfinite(g[tt, jj]), g[tt, jj] < 0, False)
    lo10, hi10 = np.percentile(r20, [10, 90])
    months = pd.DatetimeIndex(dates[tt]).strftime('%Y-%m')
    a_diff = r20[~fl].mean() - r20[fl].mean(); a_p = boot_p(months, r20, fl, rng)
    left, right = (r20[fl] <= lo10).mean(), (r20[fl] >= hi10).mean()
    V = trades('V'); c_diff = V.ret[~V.flag].mean() - V.ret[V.flag].mean()
    epsV, (dfl, nfl), (dkp, nkp) = ddc(V, 'V')
    r1 = [a_diff > 0 and a_p >= 0.90, left >= 0.15 and right <= 0.10, c_diff >= 0.01, dfl < dkp]
    L += ['## R1：V4.1\n', f"事件 {len(tt)} 個，標記（g < 0）{int(fl.sum())} 個（{fl.mean():.0%}）。",
          '| 判準 | 數值 | 門檻 | 結果 |', '|---|---|---|---|',
          f"| (a) 保留 − 標記 後 20 日超額 | {a_diff:+.2%}（保留 {r20[~fl].mean():+.2%}、標記 {r20[fl].mean():+.2%}）；月份群集 P {a_p:.3f} | > 0 且 P ≥ 0.90 | {'✓' if r1[0] else '✗'} |",
          f"| (b) 標記左尾率／右尾率 | {left:.0%}／{right:.0%} | ≥ 15%／≤ 10% | {'✓' if r1[1] else '✗'} |",
          f"| (c) 含停損交易：保留 − 標記 | {c_diff:+.2%}（保留 {V.ret[~V.flag].mean():+.2%}、{int((~V.flag).sum())} 筆；標記 {V.ret[V.flag].mean():+.2%}、{int(V.flag.sum())} 筆） | ≥ +1pt | {'✓' if r1[2] else '✗'} |",
          f"| (d) 三次最大回撤期間每筆平均損益 | 標記 {dfl:+.2%}（{nfl} 筆）vs 保留 {dkp:+.2%}（{nkp} 筆） | 標記 < 保留 | {'✓' if r1[3] else '✗'} |",
          '\nV4.1 三次最大回撤：' + '；'.join(f'{a.date()}→{b.date()} {d:.1%}' for a, b, d in epsV),
          f"\n**R1 {'通過，可登記 L 系列組合回測' if all(r1) else '不通過，結案'}**"]
    # 參考：非上升定義
    fl2 = ~(np.isfinite(g[tt, jj]) & (s200[tt, jj] > 0))
    L.append(f"\n參考（不列入判準）：200 日線未向上的事件 {int(fl2.sum())} 個，後 20 日 {r20[fl2].mean():+.2%}（其餘 {r20[~fl2].mean():+.2%}）；與 g < 0 重疊 {(fl & fl2).sum() / max(fl.sum(), 1):.0%}。")

    # ---------- R2：X1 ----------
    X = trades('X'); thi, tlo = X.ret.quantile(0.9), X.ret.quantile(0.1)
    a2 = X.ret[~X.flag].mean() - X.ret[X.flag].mean(); p2 = boot_p(X.m.values, X.ret.values, X.flag.values, rng)
    bot, top = (X.ret[X.flag] <= tlo).mean(), (X.ret[X.flag] >= thi).mean()
    epsX, (xf, nxf), (xk, nxk) = ddc(X, 'X')
    r2 = [a2 > 0 and p2 >= 0.90, bot >= 0.15 and top <= 0.10, xf < xk]
    L += ['\n## R2：X1\n', f"交易 {len(X)} 筆，標記 {int(X.flag.sum())} 筆（{X.flag.mean():.0%}）。",
          '| 判準 | 數值 | 門檻 | 結果 |', '|---|---|---|---|',
          f"| (a) 保留 − 標記 平均報酬（含停損） | {a2:+.2%}（保留 {X.ret[~X.flag].mean():+.2%}、標記 {X.ret[X.flag].mean():+.2%}）；月份群集 P {p2:.3f} | > 0 且 P ≥ 0.90 | {'✓' if r2[0] else '✗'} |",
          f"| (b) 標記落在最差／最好 10% | {bot:.0%}／{top:.0%} | ≥ 15%／≤ 10% | {'✓' if r2[1] else '✗'} |",
          f"| (c) 三次最大回撤期間每筆平均損益 | 標記 {xf:+.2%}（{nxf} 筆）vs 保留 {xk:+.2%}（{nxk} 筆） | 標記 < 保留 | {'✓' if r2[2] else '✗'} |",
          '\nX1 三次最大回撤：' + '；'.join(f'{a.date()}→{b.date()} {d:.1%}' for a, b, d in epsX),
          f"\n**R2 {'通過，可登記 L 系列組合回測' if all(r2) else '不通過，結案'}**",
          f"\n參考（不列入判準）：200 日線未向上的 X1 交易 {int(X.nonup.sum())} 筆，平均 {X.ret[X.nonup].mean():+.2%}（其餘 {X.ret[~X.nonup].mean():+.2%}）。"]
    head = f"R1（V4.1）{'通過' if all(r1) else '不通過'}；R2（X1）{'通過' if all(r2) else '不通過'}"
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
