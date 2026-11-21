# M-153：V4.1 突破事件中「三關進場法」能否區分成功與假突破（觀察式，探索集：奇數代碼、2005–15）。定義見 registry_M.md M-153。
# 產出：lab/cache/M153/plot.json（作圖用彙總，不入庫）＋ summary（run.py 寫入 lab/results/M153/）。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals

START = pd.Timestamp('2005-01-01'); END = pd.Timestamp('2015-12-31'); H = 20; COST = 0.0077
OUT = LAB / 'cache' / 'M153'


def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def stock_ind(c, h, l, v):
    """單一股票、自身有效日序列上的指標（只用當日以前資料）。"""
    s = pd.Series(c)
    ma = {n: s.rolling(n).mean().to_numpy() for n in (7, 25, 99)}
    d = np.diff(c, prepend=np.nan); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = 0.0
    au, ad = wilder(up, 14), wilder(dn, 14)
    with np.errstate(all='ignore'):
        rsi = 100 - 100 / (1 + au / ad)
    rsi[:14] = np.nan
    hh = pd.Series(h).rolling(9).max().to_numpy(); ll = pd.Series(l).rolling(9).min().to_numpy()
    with np.errstate(all='ignore'):
        rsv = np.where(hh > ll, (c - ll) / (hh - ll) * 100, 50.0)
    K = np.full(len(c), np.nan); D = np.full(len(c), np.nan); k = dd = 50.0
    for i in range(len(c)):
        if np.isnan(hh[i]): continue
        k = k * 2 / 3 + rsv[i] / 3; dd = dd * 2 / 3 + k / 3; K[i] = k; D[i] = dd
    e12 = s.ewm(span=12, adjust=False).mean(); e26 = s.ewm(span=26, adjust=False).mean()
    dif = (e12 - e26).to_numpy(); sig = pd.Series(dif).ewm(span=9, adjust=False).mean().to_numpy(); hist = dif - sig
    dif[:34] = np.nan; hist[:34] = np.nan
    vma = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    with np.errstate(all='ignore'):
        vr = v / vma
    return ma, rsi, K, D, dif, hist, vr


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, signal = dbg['U'], dbg['signal']
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); V = np.asarray(P['V'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CEW = np.cumprod(1 + ew)                                   # 可交易池等權累積指數
    sid = P['meta']['stock_id'].astype(str); odd = (sid.str[-1].astype(int) % 2 == 1).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1

    tt, jj = np.nonzero(signal)
    ok = (tt >= t_lo) & (tt + H <= t_hi) & odd[jj]; tt, jj = tt[ok], jj[ok]
    rows = []
    for j in np.unique(jj):
        ii = np.flatnonzero(VALID[:, j]); pos = {t: k for k, t in enumerate(ii)}
        c, h, l, v = CA[ii, j], HA[ii, j], LA[ii, j], V[ii, j]
        ma, rsi, K, D, dif, hist, vr = stock_ind(c, h, l, v)
        for t in tt[jj == j]:
            k = pos.get(t)
            if k is None or k < 100: continue
            m7, m25, m99 = ma[7][k], ma[25][k], ma[99][k]
            cross = [(K[i] > D[i]) and (K[i - 1] <= D[i - 1]) and K[i] >= 50 for i in (k, k - 1, k - 2)]
            r = dict(t=int(t), j=int(j), year=int(dates[t].year),
                     c_ma7=c[k] / m7 - 1, c_ma25=c[k] / m25 - 1, c_ma99=c[k] / m99 - 1,
                     rsi=rsi[k], K=K[k], D=D[k], vr=vr[k], dif=dif[k] / c[k], hist=hist[k] / c[k],
                     a_above=bool(c[k] > m7 and c[k] > m25 and c[k] > m99), a_order=bool(m7 > m25 > m99),
                     a_rise=bool(ma[7][k] > ma[7][k - 1] and ma[25][k] > ma[25][k - 1] and ma[99][k] > ma[99][k - 1]),
                     b_rsi=bool(rsi[k] >= 50 and rsi[k] > rsi[k - 1]), b_kd=bool(any(cross) and K[k] > D[k]),
                     b_kd_strict=bool(cross[0]), c_vol=bool(vr[k] > 1), c_dif=bool(dif[k] > 0), c_hist=bool(hist[k] > 0),
                     c_hist_strict=bool(hist[k] > 0 and hist[k - 1] <= 0))
            r['ex20'] = float((Cf[t + H, j] / Cf[t, j] - 1) - (CEW[t + H] / CEW[t] - 1))
            ent = OA[t + 1, j]
            r['trade'] = float(Cf[t + H, j] / ent - 1 - COST) if np.isfinite(ent) and ent > 0 else np.nan
            rows.append(r)
    X = pd.DataFrame(rows)
    X['G1'] = X.a_above & X.a_order & X.a_rise
    X['G2'] = X.b_rsi & X.b_kd
    X['G3'] = X.c_vol & X.c_dif & X.c_hist
    X['ALL'] = X.G1 & X.G2 & X.G3
    X['ALL_strict'] = X.G1 & X.b_rsi & X.b_kd_strict & X.c_vol & X.c_dif & X.c_hist_strict
    X['ngate'] = X.G1.astype(int) + X.G2 + X.G3
    X['grp'] = np.where(X.ex20 >= 0.05, '成功', np.where(X.ex20 <= -0.05, '假突破', '中間'))

    def stat(d):
        e = d.ex20
        return dict(n=int(len(d)), mean=float(e.mean()) if len(d) else np.nan, med=float(e.median()) if len(d) else np.nan,
                    win=float((e > 0).mean()) if len(d) else np.nan, succ=float((e >= 0.05).mean()) if len(d) else np.nan,
                    fake=float((e <= -0.05).mean()) if len(d) else np.nan, lt=float((e <= -0.10).mean()) if len(d) else np.nan,
                    rt=float((e >= 0.15).mean()) if len(d) else np.nan, sd=float(e.std()) if len(d) > 1 else np.nan, trade=float(d.trade.mean()) if len(d) else np.nan)
    conds = [('a_above', '1a 收盤站上三線'), ('a_order', '1b MA7>MA25>MA99'), ('a_rise', '1c 三線皆上揚'), ('G1', '第一關（全部）'),
             ('b_rsi', '2a RSI≥50 且上升'), ('b_kd', '2b KD 50 以上金叉（3 日內）'), ('G2', '第二關（全部）'),
             ('c_vol', '3a 量 > 20 日均量'), ('c_dif', '3b DIF > 0'), ('c_hist', '3c 柱狀體 > 0'), ('G3', '第三關（全部）'),
             ('ALL', '三關全過'), ('b_kd_strict', '嚴格 2b：KD 當日金叉'), ('c_hist_strict', '嚴格 3c：柱狀體當日翻紅'), ('ALL_strict', '三關全過（嚴格）')]
    tab = []
    for key, name in conds:
        a, b = stat(X[X[key]]), stat(X[~X[key]])
        tw = (a['mean'] - b['mean']) / np.sqrt(a['sd'] ** 2 / a['n'] + b['sd'] ** 2 / b['n']) if a['n'] > 1 and b['n'] > 1 else np.nan
        tab.append(dict(key=key, name=name, rate=float(X[key].mean()), yes=a, no=b, t=float(tw)))
    base = stat(X)
    gates = [dict(k=int(g), **stat(X[X.ngate == g])) for g in range(4)]
    inds = [('c_ma7', '收盤距 MA7'), ('c_ma25', '收盤距 MA25'), ('c_ma99', '收盤距 MA99'), ('rsi', 'RSI14'), ('K', 'K 值'),
            ('D', 'D 值'), ('vr', '量比（÷20 日均量）'), ('dif', 'DIF ÷ 收盤'), ('hist', '柱狀體 ÷ 收盤')]
    dist = []
    for key, name in inds:
        dd = {}
        for g in ('成功', '中間', '假突破'):
            x = X.loc[X.grp == g, key].dropna()
            dd[g] = [float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)]
        dist.append(dict(key=key, name=name, q=dd))
    grp_n = X.grp.value_counts().to_dict()
    hist_ex = np.histogram(X.ex20.clip(-0.3, 0.5), bins=np.arange(-0.3, 0.52, 0.02))[0].tolist()
    by_year = X.groupby('year').agg(n=('ex20', 'size'), all_rate=('ALL', 'mean'), g1=('G1', 'mean'), mean=('ex20', 'mean')).reset_index()
    pass_rate_grp = {g: {k: float(X.loc[X.grp == g, k].mean()) for k, _ in conds} for g in ('成功', '中間', '假突破')}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plot.json').write_text(json.dumps(dict(tab=tab, base=base, gates=gates, dist=dist, grp_n=grp_n, hist_ex=hist_ex,
                                                   by_year=by_year.round(4).to_dict('records'), pass_grp=pass_rate_grp),
                                              ensure_ascii=False, default=float), encoding='utf-8')

    pc = lambda x: '—' if not np.isfinite(x) else f'{x:+.2%}'
    pr = lambda x: '—' if not np.isfinite(x) else f'{x:.0%}'
    L = ['# M-153 結果：V4.1 突破事件 × 三關進場法（觀察式，探索集）\n',
         f"奇數代碼、2005–15。事件 {base['n']}；成功（後 20 日超額 ≥ +5%）{grp_n.get('成功', 0)}、假突破（≤ −5%）{grp_n.get('假突破', 0)}、中間 {grp_n.get('中間', 0)}。"
         f"全部：後 20 日超額平均 {pc(base['mean'])}、中位數 {pc(base['med'])}、勝率 {pr(base['win'])}；實際進場版平均 {pc(base['trade'])}。\n",
         '## 各條件：通過 vs 未通過\n',
         '| 條件 | 通過率 | 差 t | 通過 n | 平均超額 | 勝率 | 成功率 | 假突破率 | 左尾 ≤−10% | 右尾 ≥+15% | 進場版 | 未通過 n | 平均超額 | 勝率 | 成功率 | 假突破率 | 左尾 | 右尾 | 進場版 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in tab:
        a, b = r['yes'], r['no']
        L.append(f"| {r['name']} | {r['rate']:.0%} | {r['t']:+.1f} | {a['n']} | {pc(a['mean'])} | {pr(a['win'])} | {pr(a['succ'])} | {pr(a['fake'])} | {pr(a['lt'])} | {pr(a['rt'])} | {pc(a['trade'])} "
                 f"| {b['n']} | {pc(b['mean'])} | {pr(b['win'])} | {pr(b['succ'])} | {pr(b['fake'])} | {pr(b['lt'])} | {pr(b['rt'])} | {pc(b['trade'])} |")
    L += ['\n## 依通過關數\n', '| 通過關數 | n | 平均超額 | 中位數 | 勝率 | 成功率 | 假突破率 | 左尾 | 右尾 |', '|---|---|---|---|---|---|---|---|---|']
    for g in gates:
        L.append(f"| {g['k']} | {g['n']} | {pc(g['mean'])} | {pc(g['med'])} | {pr(g['win'])} | {pr(g['succ'])} | {pr(g['fake'])} | {pr(g['lt'])} | {pr(g['rt'])} |")
    L += ['\n## 成功組 vs 假突破組：指標分布（25%／中位數／75%）\n', '| 指標 | 成功 | 中間 | 假突破 |', '|---|---|---|---|']
    for d in dist:
        f = lambda q: ' ／ '.join(f'{x:.3g}' for x in q[1:4])
        L.append(f"| {d['name']} | {f(d['q']['成功'])} | {f(d['q']['中間'])} | {f(d['q']['假突破'])} |")
    L += ['\n## 各組的條件通過率\n', '| 條件 | 成功 | 中間 | 假突破 |', '|---|---|---|---|']
    for k, nm in conds:
        L.append(f"| {nm} | {pass_rate_grp['成功'][k]:.0%} | {pass_rate_grp['中間'][k]:.0%} | {pass_rate_grp['假突破'][k]:.0%} |")
    return dict(summary='\n'.join(L), headline='三關進場法在 V4.1 事件的描述（見私人網頁）', n_variants=0)
