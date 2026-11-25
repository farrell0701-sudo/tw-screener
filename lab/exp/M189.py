# M-189：X1／V4.1 vs 營收研究用組合的報酬拆解（描述性）。定義見 registry_M.md M-189。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M180, M185
from L128 import exit_of

COST = 0.0077; SLIP = 0.0015
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def rev_portfolio(P, F, U, H=120, stop=0.15, k3=2):
    dates = P['dates']; T, N = P['C'].shape; keys = P['meta']['security_key'].astype(str).to_numpy()
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); AMT = np.asarray(P['AMT'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    R6 = M180.r6_factors()['R6 營收短長比']; jm = pd.Index(R6.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2004-07-01') and dates.get_loc(d) + 2 < T]
    cache = {}
    tot = np.zeros(T); cnt = np.zeros(T); expo = np.zeros(T); live = np.zeros(T); trades = []
    for t in samp:
        m = dates[t].to_period('M') - 1
        if m not in R6.index: continue
        s0 = dates.searchsorted(dates[t].to_period('M').start_time); t0 = s0 - 1
        if t - s0 < 2: continue
        v = R6.loc[m].to_numpy(float); x = np.where(jm >= 0, v[np.clip(jm, 0, None)], np.nan)
        jj = np.nonzero(U[t] & np.isfinite(x))[0]
        if len(jj) < 100: continue
        top = jj[pd.Series(x[jj]).rank(pct=True).to_numpy() > 0.8]
        top = top[VALID[t + 1, top] & np.isfinite(OA[t + 1, top]) & (OA[t + 1, top] > 0)]
        pre = CX[t - 1, top] - CX[t0, top]
        q = pd.qcut(pd.Series(pre).rank(method='first'), 5, labels=False).to_numpy(); jj = top[q >= 1]
        e_ = min(t + 1 + H, T); n = e_ - (t + 1)
        if n < 2 or len(jj) == 0: continue
        c = pd.DataFrame(np.where(VALID[t + 1:e_][:, jj], CA[t + 1:e_][:, jj], np.nan)).ffill().to_numpy()
        px = OA[t + 1, jj] * (1 + SLIP)
        val = np.empty((n, len(jj))); alive = np.zeros((n, len(jj)), bool); info = []
        for qq, j in enumerate(jj):
            s = cache.get(j) or cache.setdefault(j, M185.Stock(j, CA, OA, HA, LA, AMT, VALID))
            ex = exit_of(s, t, H, stop, k3)
            if ex is None: val[:, qq] = c[:, qq] / px[qq]; alive[:, qq] = True; info.append(None); continue
            _, xc, xp, why = ex; k = min(xc - (t + 1), n - 1)
            val[:k, qq] = c[:k, qq] / px[qq]; val[k:, qq] = xp / px[qq]; alive[:k, qq] = True
            info.append((xp / px[qq] * (1 - COST + SLIP) - 1, k + 1))
        val = val * (1 - COST + SLIP)
        Vc = np.nanmean(val, axis=1); rc = np.r_[Vc[0] - 1, Vc[1:] / Vc[:-1] - 1]
        inv = np.nansum(np.where(alive, val, 0), axis=1) / len(jj) / Vc
        idx = np.arange(t + 1, e_); tot[idx] += rc; cnt[idx] += 1; expo[idx] += inv; live[idx] += alive.sum(axis=1)
        for z in info:
            if z is not None: trades.append((t, z[0], z[1], len(jj)))
    r = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0), index=dates)
    ex = pd.Series(np.where(cnt > 0, expo / np.maximum(cnt, 1), np.nan), index=dates)
    X = pd.DataFrame(trades, columns=['t', 'ret', 'hold', 'n'])
    # 部位權重：每組占 1/存續組數（以進場次日的 cnt），組內等權
    X['w'] = 1.0 / (np.maximum(cnt[X.t.to_numpy() + 1], 1) * X.n)
    X['date'] = dates[X.t.to_numpy()]
    return r, ex, X, pd.Series(live, index=dates)


def tstats(ret, w, lo, hi, dates_, years):
    m = (dates_ >= lo) & (dates_ <= hi); r = ret[m]; ww = w[m]
    win = r[r > 0]; loss = r[r <= 0]
    top = r.sort_values(ascending=False); top = top.iloc[:max(1, len(top) // 10)]
    net = r.sum()
    return dict(n=len(r), mean=r.mean(), med=r.median(), win=(r > 0).mean(), payoff=win.mean() / -loss.mean() if len(loss) else np.nan,
                tail=top.sum() / net if net > 0 else np.nan, turns=ww.sum() / years, wmean=(ww * r).sum() / ww.sum())


def pstats(r, ex, lo, hi):
    r = r.loc[lo:hi]; n = len(r)
    g = (1 + r).prod() ** (245 / n) - 1; a = r.mean() * 245; vol = r.std() * np.sqrt(245)
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return dict(geo=g, ari=a, drag=a - g, vol=vol, dd=dd, expo=ex.loc[lo:hi].mean())


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); U = S_['_dbg']['U']; del S_
    tx = P['index']['TAIEX'].reindex(dates).ffill() if 'TAIEX' in P['index'] else None
    if tx is None:
        d = pd.read_parquet('/home/claude/tw-data-v1.1/derived/index.parquet'); d = d[d.index_id == 'TAIEX']
        tx = pd.Series(d.price_index.to_numpy(), index=pd.to_datetime(d.date)).reindex(dates).ffill()
    bear = (tx < tx.rolling(200).mean()).shift(1).fillna(False).to_numpy().astype(bool)
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): Rr = Cf / np.vstack([np.full((1, Cf.shape[1]), np.nan), Cf[:-1]]) - 1
    Ush = np.vstack([np.zeros((1, Cf.shape[1]), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = pd.Series(np.nan_to_num(np.nanmean(np.where(Ush & VALID, Rr, np.nan), axis=1)), index=dates)
    strat = {}
    for nm, k3 in (('營收 主版本（剔Q1＋③）', 2), ('營收 R1（剔Q1、無③）', None)):
        r, ex, X, lv = rev_portfolio(P, F, U, k3=k3); strat[nm] = ('rev', r, ex, X, lv)
    B = sim.run_paths('fix', 100, sig='fix')
    for kind, nm in (('X', 'X1'), ('V', 'V4.1')):
        paths = B[kind]; strat[nm] = ('sim', paths)
    L = ['# M-189 結果：X1／V4.1 vs 營收研究用組合 報酬拆解（全部代碼；X1／V4.1 取 100 路徑中位數）\n']
    for sn, lo, hi in SEGS:
        yrs = ((min(pd.Timestamp(hi), dates[-1]) - pd.Timestamp(lo)).days + 1) / 365.25
        bm = bear[(dates >= lo) & (dates <= hi)]
        ewseg = ew.loc[lo:hi]; avoided = (1 + ewseg[bm]).prod() - 1
        L += [f'## {sn}\n', '| 策略 | 幾何年化 | 算術年化 | 波動拖累 | 年化波動 | 最大回撤 | 資金使用率 | 空頭日使用率 | 多頭日使用率 | 平均持股 | 周轉（輪／年） | 平均持有日 | 單筆平均 | 單筆中位 | 勝率 | 盈虧比 | 前 10% 交易占淨利 | 周轉×加權單筆 |',
              '|' + '---|' * 18]
        for nm, s in strat.items():
            if s[0] == 'rev':
                _, r, ex, X, lv = s
                p = pstats(r, ex, lo, hi); exs = ex.loc[lo:hi]
                tb, tu = exs[bm].mean(), exs[~bm].mean()
                ts = tstats(X.ret, X.w, lo, hi, X.date, yrs)
                hold = X[(X.date >= lo) & (X.date <= hi)].hold.mean(); nh = lv.loc[lo:hi].mean()
            else:
                rows = []
                for (r, tr, ex) in s[1]:
                    p = pstats(r, ex, lo, hi); exs = ex.reindex(dates).loc[lo:hi]
                    T_ = pd.DataFrame(tr, columns=['code', 'd0', 'd1', 'w', 'ret', 'hold', 'why', 'st', 'lk'])
                    ts = tstats(T_.ret, T_.w, lo, hi, pd.to_datetime(T_.d0), yrs)
                    T2 = T_[(pd.to_datetime(T_.d0) >= lo) & (pd.to_datetime(T_.d0) <= hi)]
                    rows.append({**p, **ts, 'tb': exs[bm].mean(), 'tu': exs[~bm].mean(), 'hold': T2.hold.mean(), 'nh': np.nan})
                D = pd.DataFrame(rows).median(); p = D; ts = D; tb, tu, hold, nh = D.tb, D.tu, D.hold, (6 if nm == 'X1' else 4)
            L.append(f"| {nm} | {p['geo']:+.1%} | {p['ari']:+.1%} | {p['drag']:.1%} | {p['vol']:.1%} | {p['dd']:.1%} | {p['expo']:.0%} | {tb:.0%} | {tu:.0%} | {nh:.0f} | "
                     f"{ts['turns']:.1f} | {hold:.0f} | {ts['mean']:+.2%} | {ts['med']:+.2%} | {ts['win']:.0%} | {ts['payoff']:.2f} | {ts['tail']:.0%} | {ts['turns'] * ts['wmean']:+.1%} |")
        L += ['', f"加權指數在 200 日均線之下的交易日占 {bm.mean():.0%}；這些日子等權可交易池累積報酬 {avoided:+.1%}（閘門可避開的跌幅上限）。\n"]
    L.append('註：X1／V4.1 平均持股為名額上限（6／4）；營收為實際平均持有檔數。周轉＝每年進場部位占權益比例總和；「周轉×加權單筆」應接近算術年化（扣除未平倉與現金報酬差）。')
    return dict(summary='\n'.join(L), headline='報酬拆解', n_variants=0)
