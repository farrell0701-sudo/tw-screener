# L-128：L-109 第一關：R6 頭部剔除月初弱勢＋③ 保護獲利（研究用組合）。判準見 registry_L.md L-128。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
import M180, M185
from L126 import stats, SEGS

COST = 0.0077; SLIP = 0.0015
VARS = {'H120／S15／③k2（主）': (120, 0.15, 2), 'H90': (90, 0.15, 2), 'H150': (150, 0.15, 2), 'S12': (120, 0.12, 2), 'S18': (120, 0.18, 2),
        '③k1': (120, 0.15, 1), '③k3': (120, 0.15, 3)}
REFS = {'R0 不剔Q1、無③': (120, 0.15, None, False), 'R1 剔Q1、無③': (120, 0.15, None, True)}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def exit_of(S, t, H, stop, k3):
    """回傳 (進場有效日序列位置 e, 出場日曆索引, 出場價, 原因)。"""
    v = S.v; e = np.searchsorted(v, t + 1)
    if e >= len(v) or v[e] != t + 1: return None
    end = np.searchsorted(v, t + H, side='right') - 1
    if end < e: return None
    ent = S.o[e]; c = S.c
    trig = None
    if k3 is not None:
        runmax = c[e]; lastnh = None; cands = []
        for i in range(e + 1, end + 1):
            if c[i] > runmax: runmax = c[i]; lastnh = i
            if lastnh is not None and 1 <= i - lastnh <= 10 and S.down[i] and np.isfinite(S.a95[i]) and S.a[i] >= S.a95[i]: cands.append(i)
        for b in cands:
            run = 0
            for i in range(b + 1, end + 1):
                if c[i] < S.l[b]:
                    run += 1
                    if run == k3:
                        if trig is None or i < trig: trig = i
                        break
                else: break
    st = next((i for i in range(e, end + 1) if c[i] < ent * (1 - stop)), None)
    cand = [(x, why) for x, why in ((st, 'stop'), (trig, '③')) if x is not None]
    if cand:
        x, why = min(cand)
        if x + 1 <= end: return e, v[x + 1], S.o[x + 1], why
        return e, v[x], c[x], why
    return e, v[end], c[end], 'time'


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); U = S_['_dbg']['U']; del S_
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); AMT = np.asarray(P['AMT'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    ewr = pd.Series(ew, index=dates)
    R6 = M180.r6_factors()['R6 營收短長比']; jm = pd.Index(R6.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2004-07-01') and dates.get_loc(d) + 2 < T]
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    mem_all, mem_x = {}, {}
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
        q = pd.qcut(pd.Series(pre).rank(method='first'), 5, labels=False).to_numpy()
        mem_all[t] = top; mem_x[t] = top[q >= 1]
    cache = {}
    def stock(j):
        s = cache.get(j)
        if s is None: s = cache[j] = M185.Stock(j, CA, OA, HA, LA, AMT, VALID)
        return s

    def build(H, stop, k3, excl=True):
        mem = mem_x if excl else mem_all
        tot = np.zeros(T); cnt = np.zeros(T); live = np.zeros(T); trades = []
        for t, jj in mem.items():
            e_ = min(t + 1 + H, T); n = e_ - (t + 1)
            if n < 2 or len(jj) == 0: continue
            c = pd.DataFrame(np.where(VALID[t + 1:e_][:, jj], CA[t + 1:e_][:, jj], np.nan)).ffill().to_numpy()
            px = OA[t + 1, jj] * (1 + SLIP)
            val = np.empty((n, len(jj))); alive = np.zeros((n, len(jj)), bool)
            for q, j in enumerate(jj):
                ex = exit_of(stock(j), t, H, stop, k3)
                if ex is None:
                    val[:, q] = c[:, q] / px[q]; alive[:, q] = True; continue
                _, xc, xp, why = ex; k = min(xc - (t + 1), n - 1)
                val[:k, q] = c[:k, q] / px[q]; val[k:, q] = xp / px[q]; alive[:k + 1, q] = True
                trades.append((t, j, xp / px[q] * (1 - COST + SLIP) - 1, k + 1, why))
            val = val * (1 - COST + SLIP)
            Vc = np.nanmean(val, axis=1); rc = np.r_[Vc[0] - 1, Vc[1:] / Vc[:-1] - 1]
            idx = np.arange(t + 1, e_); tot[idx] += rc; cnt[idx] += 1; live[idx] += alive.sum(axis=1)
        r = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0), index=dates)
        return r, pd.DataFrame(trades, columns=['t', 'j', 'ret', 'hold', 'why']), pd.Series(live, index=dates)

    OUT = {nm: build(H, s, k) for nm, (H, s, k) in VARS.items()}
    REF = {nm: build(H, s, k, ex) for nm, (H, s, k, ex) in REFS.items()}
    log('## 單獨表現（年化／最大回撤／Calmar／相對 TR 年化超額）\n')
    log('| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |\n|---|---|---|---|---|---|')
    ex = {}
    for nm, (r, X, lv) in list(OUT.items()) + list(REF.items()):
        cells = []
        for sn, lo, hi in SEGS:
            a, dd, cm, x, _ = stats(r, tr, lo, hi); cells.append(f'{a:+.1%}／{dd:.1%}／{cm:.2f}／{x:+.1%}')
            ex[(nm, sn)] = x
        log(f'| {nm} | ' + ' | '.join(cells) + ' |')
    log('| 加權報酬指數 TR | ' + ' | '.join(f"{stats(tr, tr, lo, hi)[0]:+.1%}／{stats(tr, tr, lo, hi)[1]:.1%}／{stats(tr, tr, lo, hi)[2]:.2f}／—" for _, lo, hi in SEGS) + ' |')
    log('| 等權可交易池 | ' + ' | '.join(f"{stats(ewr, tr, lo, hi)[0]:+.1%}／{stats(ewr, tr, lo, hi)[1]:.1%}／{stats(ewr, tr, lo, hi)[2]:.2f}／{stats(ewr, tr, lo, hi)[3]:+.1%}" for _, lo, hi in SEGS) + ' |')
    log('\n相對等權可交易池年化超額（只報告）：' + '；'.join(f"{nm}：" + '｜'.join(f"{stats(OUT[nm][0] if nm in OUT else REF[nm][0], ewr, lo, hi)[3]:+.1%}" for sn, lo, hi in SEGS[:4]) for nm in ['H120／S15／③k2（主）', 'R0 不剔Q1、無③', 'R1 剔Q1、無③']) + '（2005–15｜2016–20｜2021–26｜2016–26）')
    r, X, lv = OUT['H120／S15／③k2（主）']
    X['date'] = dates[X.t]; X5 = X[X.date >= '2005-01-01']
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    log(f"\n主版本：交易 {len(X5)} 筆（每年約 {len(X5) / yrs:.0f} 筆）；平均同時持有 {lv.loc['2005':].mean():.0f} 檔（最多 {lv.loc['2005':].max():.0f}）；"
        f"出場原因 停損 {(X5.why == 'stop').mean():.0%}、③ {(X5.why == '③').mean():.0%}、到期 {(X5.why == 'time').mean():.0%}；每筆平均 {X5.ret.mean():+.2%}、中位數 {X5.ret.median():+.2%}、勝率 {(X5.ret > 0).mean():.0%}、平均持有 {X5.hold.mean():.0f} 日\n")
    ry = (1 + r).groupby(r.index.year).prod() - 1; ty = (1 + tr).groupby(tr.index.year).prod() - 1
    ey = (1 + ewr).groupby(ewr.index.year).prod() - 1
    B = sim.run_paths('fix', 100, sig='fix'); bpm = E.path_mean([sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(100)])
    by = (1 + bpm).groupby(bpm.index.year).prod() - 1
    log('## 每年報酬\n'); log('| 年 | 營收子策略 S | 加權報酬指數 | S − TR | 等權可交易池 | 新基準合併版 |\n|---|---|---|---|---|---|')
    for y in ry.index:
        if y < 2005: continue
        log(f"| {y} | {ry[y]:+.1%} | {ty[y]:+.1%} | {ry[y] - ty[y]:+.1%} | {ey.get(y, np.nan):+.1%} | {by.get(y, np.nan):+.1%} |")
    a = pd.concat([r, bpm], axis=1, join='inner').loc['2005':]; a.columns = ['S', 'B']
    corr = a.S.corr(a.B); mon = (1 + a).groupby(a.index.to_period('M')).prod() - 1; worst = mon.B <= mon.B.quantile(0.1)
    log(f"\n與新基準合併版日報酬相關係數：{corr:.2f}")
    log(f"新基準最差 10% 月份（{worst.sum()} 個月）：新基準平均 {mon.B[worst].mean():+.1%}、S 平均 {mon.S[worst].mean():+.1%}（第二關預覽，只報告）\n")
    log('## 判準（L-109 第一關，事前固定）\n')
    ok = True
    for nm in VARS:
        c = ex[(nm, '2005–15')] > 0 and ex[(nm, '2016–26')] > 0; ok = ok and c
        log(f"- {nm}：2005–15 超額 {ex[(nm, '2005–15')]:+.1%}、2016–26 超額 {ex[(nm, '2016–26')]:+.1%} → {'✓' if c else '✗'}")
    ntr = len(X5) >= 100; ok = ok and ntr
    log(f"- 交易數 ≥ 100：{'✓' if ntr else '✗'}（{len(X5)}）")
    log(f"\n**結論：{'第一關通過（概念成立）' if ok else '第一關不通過'}**\n")
    head = f"L-128 R6 頭部剔 Q1＋③：第一關{'通過' if ok else '不通過'}（主版本超額 {ex[('H120／S15／③k2（主）', '2005–15')]:+.1%}｜{ex[('H120／S15／③k2（主）', '2016–26')]:+.1%}；相關 {corr:.2f}）"
    return dict(summary='# L-128 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: v[0] for k, v in OUT.items()}})
