# M-190：生命週期的兩個預測（P1 營收池→強勢池；P2 高波動 × 融資增量）。定義見 registry_M.md M-190。
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M165, M120

END = pd.Timestamp('2024-11-13')
PARITY = int(os.environ.get('M190_PARITY', '1'))
ERAS = ('2005–15', '2016–24/11')


def margin_panel(dates, meta):
    d = M120.TWD / 'staged' / 'TaiwanStockMarginPurchaseShortSale'
    fs = [pd.read_parquet(d / f'{y}.parquet', columns=['date', 'stock_id', 'MarginPurchaseTodayBalance']) for y in range(2004, 2025)]
    x = pd.concat(fs); x['date'] = pd.to_datetime(x['date']); x['stock_id'] = x['stock_id'].astype(str)
    W = x.pivot_table(index='date', columns='stock_id', values='MarginPurchaseTodayBalance', aggfunc='last').reindex(dates)
    sid = meta['stock_id'].astype(str).to_numpy(); j = pd.Index(W.columns).get_indexer(sid)
    M = np.where(j >= 0, W.to_numpy(float)[:, np.clip(j, 0, None)], np.nan)
    return M


def stats(x):
    return dict(L=x.L.mean(), Rt=x.Rt.mean(), DD=x.DD.mean(), mu=x.e60.mean(), n=len(x))


def fmt(s):
    return f"{s['L']:.0%}｜{s['Rt']:.0%}｜{s['DD']:.0%}｜{s['mu']:+.1%}（n={s['n']}）" if s['n'] >= 30 else f"—（n={s['n']}）"


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    par = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    C = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    Rz = np.where(VALID & ~np.isnan(R), R, 0.0)
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, Rz, np.nan), axis=1))
    CX = np.cumsum(Rz - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    ATRP = np.asarray(F['ATR'], np.float64) / C
    PC = np.vstack([np.zeros((1, N)), np.cumsum(pool, axis=0)])        # PC[t] = pool 在 [0, t) 的天數
    MG = margin_panel(dates, meta)
    SH = M165.shares_table(dates); shj = pd.Index(SH.columns.astype(str)).get_indexer(keys)
    samp = sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0])
    rows1, rows2 = [], []
    for k, (m, (t, tp)) in enumerate(samp):
        if dates[t] < pd.Timestamp('2005-01-01') or dates[t] > END or t + 61 >= T or k < 3: continue
        era = ERAS[0] if dates[t] <= pd.Timestamp('2015-12-31') else ERAS[1]
        Ut = U[t]; e60 = CX[t + 60] - CX[t]; q10, q90 = np.nanpercentile(e60[Ut], [10, 90])
        with np.errstate(all='ignore'): dd = np.nanmin(C[t + 1:t + 61], axis=0) / C[t] - 1
        frac = (PC[t - 20] - PC[t - 60]) / 40.0
        fromA = A[samp[k - 1][1][0]] | A[samp[k - 2][1][0]] | A[samp[k - 3][1][0]]
        idx = np.nonzero(pool[t] & Ut)[0]
        atr_hi = pd.Series(ATRP[tp, idx]).rank(pct=True).to_numpy() > 0.5
        sh = np.where(shj >= 0, SH.loc[m].to_numpy(float)[np.clip(shj, 0, None)], np.nan) if m in SH.index else np.full(N, np.nan)
        dmr = {L: (MG[tp] - MG[tp - L]) * 1000 / sh for L in (20, 10, 40)}
        for name, base in (('強勢池', idx), ('營收池', np.nonzero(A[t] & Ut)[0])):
            hi = atr_hi if name == '強勢池' else pd.Series(ATRP[tp, base]).rank(pct=True).to_numpy() > 0.5
            terc = {}
            for L in (20, 10, 40):
                v = dmr[L][base]; r = np.full(len(base), np.nan)
                for h in (True, False):
                    s = (hi == h) & np.isfinite(v)
                    if s.sum() >= 6: r[s] = pd.Series(v[s]).rank(pct=True).to_numpy()
                terc[L] = np.where(np.isnan(r), '', np.where(r > 2 / 3, '高', np.where(r <= 1 / 3, '低', '中')))
            for ii, j in enumerate(base):
                if par[j]: rows2.append((era, name, '高波動' if hi[ii] else '低波動', terc[20][ii], terc[10][ii], terc[40][ii], e60[j], e60[j] <= q10, e60[j] >= q90, dd[j] <= -0.2))
        for ii, j in enumerate(idx):
            if not par[j]: continue
            g = ('新進自營收池' if fromA[j] else '新進非營收') if frac[j] <= 0.2 else ('老成員' if frac[j] >= 0.8 else '其他')
            rows1.append((era, g, '高波動' if atr_hi[ii] else '低波動', e60[j], e60[j] <= q10, e60[j] >= q90, dd[j] <= -0.2))
    D1 = pd.DataFrame(rows1, columns=['era', 'g', 'vol', 'e60', 'L', 'Rt', 'DD'])
    D2 = pd.DataFrame(rows2, columns=['era', 'pool', 'vol', 't20', 't10', 't40', 'e60', 'L', 'Rt', 'DD'])
    out = [f"# M-190 結果：生命週期的兩個預測（{'奇' if PARITY else '偶'}數代碼；60 日超額；格式 左尾｜右尾｜DD｜平均）\n", '## 預測一：從營收池轉進強勢池\n',
           '| 組別 | 2005–15 | 2016–24/11 |', '|---|---|---|']
    S1 = {}
    for g in ('新進自營收池', '新進非營收', '其他', '老成員'):
        S1[g] = {e: stats(D1[(D1.era == e) & (D1.g == g)]) for e in ERAS}
        out.append(f"| {g} | {fmt(S1[g][ERAS[0]])} | {fmt(S1[g][ERAS[1]])} |")
    for v in ('高波動', '低波動'):
        s = {e: stats(D1[(D1.era == e) & (D1.g == '老成員') & (D1.vol == v)]) for e in ERAS}
        out.append(f"| 老成員 × {v} | {fmt(s[ERAS[0]])} | {fmt(s[ERAS[1]])} |")
    ok1 = True
    for e in ERAS:
        a, b, o = S1['新進自營收池'][e], S1['新進非營收'][e], S1['老成員'][e]
        d = (a['Rt'] - a['L']) - (b['Rt'] - b['L'])
        c = d >= 0.05 and a['mu'] > b['mu'] and a['mu'] > o['mu']; ok1 &= c
        out.append(f"\n- {e}：淨尾差（新進自營收池 − 新進非營收）{d * 100:+.1f}；平均差 {(a['mu'] - b['mu']) * 100:+.1f}；相對老成員 {(a['mu'] - o['mu']) * 100:+.1f} → {'✓' if c else '✗'}")
    out.append(f"\n**P1：{'提名' if ok1 else '不提名'}**\n")
    out += ['## 預測二：高波動 × 融資增量（ΔMR，單位：股本的百分點）\n']
    ok2 = True
    for pname in ('強勢池', '營收池'):
        out += [f'### {pname}\n', '| 波動 | ΔMR20 三等分 | 2005–15 | 2016–24/11 |', '|---|---|---|---|']
        for v in ('高波動', '低波動'):
            for tr in ('高', '中', '低'):
                s = {e: stats(D2[(D2.era == e) & (D2.pool == pname) & (D2.vol == v) & (D2.t20 == tr)]) for e in ERAS}
                out.append(f"| {v} | {tr} | {fmt(s[ERAS[0]])} | {fmt(s[ERAS[1]])} |")
        out.append('')
        if pname != '強勢池': continue
        for e in ERAS:
            res = {}
            for L in (20, 10, 40):
                col = f't{L}'; x = D2[(D2.era == e) & (D2.pool == pname)]
                lr = lambda v, tr: x[(x.vol == v) & (x[col] == tr)].L.mean()
                res[L] = (lr('高波動', '高'), lr('高波動', '低'), lr('低波動', '高'), lr('低波動', '低'))
            h, l, h2, l2 = res[20]
            c1 = h >= 1.3 * l; c2 = (h - l) > (h2 - l2); c3 = all((res[L][0] - res[L][1]) * (h - l) > 0 for L in (10, 40))
            ok2 &= c1 and c2 and c3
            out.append(f"- {e}：高波動 左尾 高 {h:.0%} vs 低 {l:.0%}（(1) {'✓' if c1 else '✗'}）；低波動 {h2:.0%} vs {l2:.0%}（(2) {'✓' if c2 else '✗'}）；10 日 {res[10][0]:.0%} vs {res[10][1]:.0%}、40 日 {res[40][0]:.0%} vs {res[40][1]:.0%}（(3) {'✓' if c3 else '✗'}）")
        out.append('')
    out.append(f"**P2：{'提名' if ok2 else '不提名'}**")
    return dict(summary='\n'.join(out), headline=f"P1 {'提名' if ok1 else '不提名'}；P2 {'提名' if ok2 else '不提名'}", n_variants=0)
