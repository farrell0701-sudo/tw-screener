# L-126：L-109 第一關（單獨表現，概念驗證）：池外營收創高十分組研究用組合。判準見 registry_L.md L-126。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M120 import rev_factors

COST = 0.0077; SLIP = 0.0015
VARS = {'H120／S15（主）': (120, 0.15), 'H90／S15': (90, 0.15), 'H150／S15': (150, 0.15), 'H120／S12': (120, 0.12), 'H120／S18': (120, 0.18)}
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'),
        ('2016–26', '2016-01-01', '2026-12-31'), ('全期', '2005-01-01', '2026-12-31'))
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def stats(r, tr, lo, hi):
    r = r.loc[lo:hi]; tr = tr.reindex(r.index).fillna(0); n = len(r)
    eq = (1 + r).cumprod(); ann = eq.iloc[-1] ** (245 / n) - 1; dd = (eq / eq.cummax() - 1).min()
    tann = (1 + tr).prod() ** (245 / n) - 1
    return ann, dd, ann / -dd if dd < 0 else np.nan, ann - tann, tann


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; del S
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    W = rev_factors(dates)['R4 營收創高']; jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2004-07-01') and dates.get_loc(d) + 2 < T]
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    members = {}
    for t in samp:
        m = dates[t].to_period('M') - 1
        if m not in W.index: continue
        v = W.loc[m].to_numpy(float); x = np.where(jm >= 0, v[np.clip(jm, 0, None)], np.nan)
        jj = np.nonzero(U[t] & ~pool[t] & np.isfinite(x))[0]
        if len(jj) < 100: continue
        g = pd.qcut(pd.Series(x[jj]).rank(method='first').to_numpy(), 10, labels=False); top = jj[g == 9]
        top = top[VALID[t + 1, top] & np.isfinite(OA[t + 1, top]) & (OA[t + 1, top] > 0)]
        members[t] = top

    def build(H, stop):
        tot = np.zeros(T); cnt = np.zeros(T); live = np.zeros(T); trades = []
        for t, jj in members.items():
            e = min(t + 1 + H, T); idx = np.arange(t + 1, e); n = len(idx)
            if n < 2: continue
            c = np.where(VALID[t + 1:e][:, jj], CA[t + 1:e][:, jj], np.nan)
            c = pd.DataFrame(c).ffill().to_numpy()
            px = OA[t + 1, jj] * (1 + SLIP)
            hit = c < px * (1 - stop)
            k = np.where(hit.any(axis=0), hit.argmax(axis=0), n - 1)
            ar = np.arange(n)[:, None]
            val = np.where(ar <= k[None, :], c / px, (c[k, np.arange(len(jj))] / px)[None, :]) * (1 - COST + SLIP)
            Vc = val.mean(axis=1); rc = np.diff(np.r_[1.0, Vc])
            rc = np.r_[Vc[0] - 1, Vc[1:] / Vc[:-1] - 1]
            tot[idx] += rc; cnt[idx] += 1; live[idx] += (ar <= k[None, :]).sum(axis=1)
            for q, j in enumerate(jj):
                trades.append((t, j, val[k[q], q] - 1, int(k[q]) + 1, bool(hit[k[q], q])))
        r = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0), index=dates)
        return r, pd.DataFrame(trades, columns=['t', 'j', 'ret', 'hold', 'stopped']), pd.Series(live, index=dates)

    OUT = {nm: build(H, s) for nm, (H, s) in VARS.items()}
    log('## 單獨表現（年化／最大回撤／Calmar／相對 TR 年化超額）\n')
    log('| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |\n|---|---|---|---|---|---|')
    ok = True; ex = {}
    for nm, (r, X, lv) in OUT.items():
        cells = []
        for sn, lo, hi in SEGS:
            a, dd, cm, x, _ = stats(r, tr, lo, hi); cells.append(f'{a:+.1%}／{dd:.1%}／{cm:.2f}／{x:+.1%}')
            if sn in ('2005–15', '2016–26'): ex[(nm, sn)] = x
        log(f'| {nm} | ' + ' | '.join(cells) + ' |')
    tline = ['| 加權報酬指數 TR | ' + ' | '.join(f"{stats(tr, tr, lo, hi)[0]:+.1%}／{stats(tr, tr, lo, hi)[1]:.1%}／{stats(tr, tr, lo, hi)[2]:.2f}／—" for _, lo, hi in SEGS) + ' |']
    log(tline[0])
    r, X, lv = OUT['H120／S15（主）']
    X['date'] = dates[X.t]; X5 = X[X.date >= '2005-01-01']
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    log(f"\n主版本：交易 {len(X5)} 筆（每年約 {len(X5) / yrs:.0f} 筆）；平均同時持有 {lv.loc['2005':].mean():.0f} 檔（最多 {lv.loc['2005':].max():.0f}）；停損出場 {X5.stopped.mean():.0%}；每筆平均淨報酬 {X5.ret.mean():+.2%}、勝率 {(X5.ret > 0).mean():.0%}、平均持有 {X5.hold.mean():.0f} 日\n")
    # 每年
    ry = (1 + r).groupby(r.index.year).prod() - 1; ty = (1 + tr).groupby(tr.index.year).prod() - 1
    # 新基準
    B = sim.run_paths('fix', 100, sig='fix'); bpm = E.path_mean([sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(100)])
    by = (1 + bpm).groupby(bpm.index.year).prod() - 1
    log('## 每年報酬\n'); log('| 年 | 營收子策略 S | 加權報酬指數 | S − TR | 新基準合併版 |\n|---|---|---|---|---|')
    for y in ry.index:
        if y < 2005: continue
        log(f"| {y} | {ry[y]:+.1%} | {ty[y]:+.1%} | {ry[y] - ty[y]:+.1%} | {by.get(y, np.nan):+.1%} |")
    a = pd.concat([r, bpm], axis=1, join='inner').loc['2005':]; a.columns = ['S', 'B']
    corr = a.S.corr(a.B)
    mon = (1 + a).groupby(a.index.to_period('M')).prod() - 1; worst = mon.B <= mon.B.quantile(0.1)
    log(f"\n與新基準合併版日報酬相關係數：{corr:.2f}")
    log(f"新基準最差 10% 月份（{worst.sum()} 個月）：新基準平均 {mon.B[worst].mean():+.1%}、S 平均 {mon.S[worst].mean():+.1%}（第二關預覽，只報告）")
    # 與 X1 重疊（新基準第 0 條路徑）
    held = {}
    for (code, d0, d1, *_ ) in B['X'][0][1]: held.setdefault(code, []).append((d0, d1))
    ov = []
    for t, j in zip(X5.t, X5.j):
        d = dates[t + 1]; ov.append(any(d0 <= d < d1 for d0, d1 in held.get(keys[j], [])))
    log(f"S 進場時該股正被 X1 持有的比例（新基準路徑 0）：{np.mean(ov):.1%}（只報告）\n")
    # 判準
    log('## 判準（L-109 第一關，事前固定）\n')
    for nm in VARS:
        c = ex[(nm, '2005–15')] > 0 and ex[(nm, '2016–26')] > 0; ok = ok and c
        log(f"- {nm}：2005–15 超額 {ex[(nm, '2005–15')]:+.1%}、2016–26 超額 {ex[(nm, '2016–26')]:+.1%} → {'✓' if c else '✗'}")
    ntr = len(X5) >= 100
    log(f"- 交易數 ≥ 100：{'✓' if ntr else '✗'}（{len(X5)}）")
    ok = ok and ntr
    log(f"\n**結論：{'第一關通過（概念成立），可登記實用版' if ok else '第一關不通過'}**\n")
    head = f"L-126 池外營收十分組：第一關{'通過' if ok else '不通過'}（主版本超額 {ex[('H120／S15（主）', '2005–15')]:+.1%}｜{ex[('H120／S15（主）', '2016–26')]:+.1%}；相關 {corr:.2f}）"
    return dict(summary='# L-126 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: v[0] for k, v in OUT.items()}})
