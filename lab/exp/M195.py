# M-195：持有中面板——「死錢」與「崩跌」的前兆（奇數代碼；到 2024-11-13）。定義見 registry_M.md M-195。
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M120
from L140 import fixed_industry

END = pd.Timestamp('2024-11-13')
PARITY = int(os.environ.get('M195_PARITY', '1'))
ERAS = ('2005–15', '2016–24/11')
f32 = np.float32


def roll(a, w, fn, mp=None):
    r = pd.DataFrame(a).rolling(w, min_periods=mp or w)
    return getattr(r, fn)().to_numpy(f32)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']
    par = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False)
    Tr = full(dbg['Tr'], np.nan).astype(f32); RKX = full(S['RKX'], np.nan); below = S['daily']['below'].to_numpy(bool); del S, dbg
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cdf = pd.DataFrame(CA).ffill(); C = Cdf.to_numpy(); O = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); ATR = np.asarray(F['ATR'], np.float64); ATRP = ATR / C
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    Rz = np.where(VALID & ~np.isnan(R), R, 0.0)
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, Rz, np.nan), axis=1))
    CX = np.cumsum(Rz - ew[:, None], axis=0)
    tend = int(dates.searchsorted(END, side='right')) - 1; s0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))

    # ---- 模擬所有候選（奇數代碼、名次 ≤ 12、閘門開）
    trades = {}   # (s, j) -> (px, exit_px, t_exit_signal_day k)
    bench = np.zeros(T)
    for s in range(s0, tend - 1):
        if below[s]: continue
        rets = []
        for j in np.nonzero((RKX[s] <= 12) & par)[0]:
            t = s + 1; px = O[t, j]
            if not np.isfinite(px) or px <= 0: continue
            d = ATRP[s, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
            hi = px; ex = None
            for k in range(t, min(t + 250, tend)):
                cc = CA[k, j]
                if not np.isfinite(cc): continue
                hi = max(hi, cc)
                if cc < hi * (1 - d) or (hi >= px * 1.06 and cc < px * 1.005) or k - t + 1 >= 250:
                    o = O[k + 1, j]; ex = (o if np.isfinite(o) else cc, k); break
            if ex is None: continue
            trades[(s, j)] = (px, ex[0], ex[1], d); rets.append(ex[0] / px - 1)
        bench[s] = np.mean(rets) if rets else 0.0
    # bench[d]：在 d+1 進場的新候選平均報酬（訊號日 d）
    # ---- 特徵
    dT = (Tr - pd.DataFrame(Tr).shift(10).to_numpy(f32))
    ind = np.array([str(v) if isinstance(v, str) else '缺' for v in fixed_industry(meta)[0].astype(object)])
    codes, inv = np.unique(ind, return_inverse=True); K = len(codes)
    x20 = (CX - np.vstack([np.full((20, N), np.nan), CX[:-20]])).astype(f32)
    AMT = np.asarray(P['AMT'], np.float64); AMT = np.where(VALID, AMT, np.nan)
    am50 = pd.DataFrame(AMT).rolling(50, min_periods=30).mean().shift(1).to_numpy()
    dist = ((np.nan_to_num(R) < 0) & (AMT > 1.5 * am50)).astype(f32); DIST20 = roll(dist, 20, 'sum', 1); del dist, am50
    ush = (HA - np.fmax(O, CA)) / ATR; USH5 = roll(np.where(np.isfinite(ush), ush, 0), 5, 'mean', 1); del ush
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); ixhi = (ix >= ix.rolling(60).max() - 1e-9).astype(float).rolling(5, min_periods=1).max().to_numpy() > 0
    MX60 = Cdf.rolling(60, min_periods=40).max().to_numpy(); sthi = roll((C >= MX60 - 1e-9).astype(f32), 10, 'max', 1) > 0
    # 融資
    dmg = M120.TWD / 'staged' / 'TaiwanStockMarginPurchaseShortSale'
    mg = pd.concat([pd.read_parquet(dmg / f'{y}.parquet', columns=['date', 'stock_id', 'MarginPurchaseTodayBalance']) for y in range(2004, 2025)])
    mg['date'] = pd.to_datetime(mg['date']); mg['stock_id'] = mg['stock_id'].astype(str)
    W = mg.pivot_table(index='date', columns='stock_id', values='MarginPurchaseTodayBalance', aggfunc='last').reindex(dates); del mg
    jj = pd.Index(W.columns).get_indexer(meta['stock_id'].astype(str)); MG = np.where(jj >= 0, W.to_numpy(float)[:, np.clip(jj, 0, None)], np.nan); del W
    am20 = pd.DataFrame(AMT).rolling(20, min_periods=10).mean().to_numpy()
    RAWC = np.asarray(P['C'], np.float64)
    DMR = ((MG - np.vstack([np.full((20, N), np.nan), MG[:-20]])) * 1000 * RAWC / am20).astype(f32); del MG, am20
    EMA12 = Cdf.ewm(span=12, adjust=False).mean().to_numpy(); EXT12 = ((C - EMA12) / ATR).astype(f32); del EMA12
    lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
    LU5 = roll((np.nan_to_num(R) >= 0.95 * lim).astype(f32), 5, 'sum', 1)
    Md = Cdf.ewm(span=200, adjust=False).mean() - Cdf.ewm(span=201, adjust=False).mean(); SGd = Md.ewm(span=202, adjust=False).mean()
    M = Md.to_numpy(); SG = SGd.to_numpy(); d5 = (Md - Md.shift(5)).to_numpy()
    hi120 = (Md >= Md.rolling(120, min_periods=120).max() - 1e-15).to_numpy().astype(f32)
    hi60 = (Md >= Md.rolling(60, min_periods=60).max() - 1e-15).to_numpy().astype(f32); del Md, SGd
    h10 = roll(hi120, 10, 'max', 1) > 0
    e1 = (d5 < 0) & (np.vstack([np.full((1, N), np.nan), d5[:-1]]) >= 0) & np.vstack([np.zeros((1, N), bool), h10[:-1]])
    no20 = roll(hi60, 20, 'sum') == 0; cnt = pd.DataFrame(hi60).shift(20).rolling(40, min_periods=40).sum().to_numpy()
    c2 = no20 & (cnt >= 20); e2 = c2 & ~np.vstack([np.zeros((1, N), bool), c2[:-1]])
    e3 = (M < SG) & (np.vstack([np.full((1, N), np.nan), M[:-1] - SG[:-1]]) >= 0) & (M > 0)
    MACD5 = roll((e1 | e2 | e3).astype(f32), 5, 'max', 1) > 0
    del M, SG, d5, hi120, hi60, h10, e1, e2, e3, no20, cnt, c2
    # ---- 面板
    rows = []
    for (s, j), (px, xp, k_ex, dd) in trades.items():
        if (s - s0) % 5: continue
        t = s + 1; hi = px
        for dday in range(t, k_ex + 1):
            cc = CA[dday, j]
            if not np.isfinite(cc): continue
            hi = max(hi, cc)
            if dday == k_ex: break
            if dday + 20 > tend: break
            u = U[dday]
            # 相對產業（同產業 U 內其他股票）
            rows.append((dday, j, dday - t, cc / px - 1, (cc / hi - (1 - dd)) / dd, xp / cc - 1, bench[dday], CX[dday + 20, j] - CX[dday, j]))
    D = pd.DataFrame(rows, columns=['d', 'j', 'held', 'unr', 'stopd', 'rem', 'bench', 'f20'])
    dd_, jj_ = D.d.to_numpy(), D.j.to_numpy()
    # 產業相對：逐日計算有面板列的日子
    relind = np.full(len(D), np.nan)
    for dday, g in D.groupby('d').groups.items():
        u = U[dday] & np.isfinite(x20[dday]); v = np.where(u, x20[dday], 0)
        sm = np.bincount(inv, v, K); ct = np.bincount(inv, u.astype(float), K)
        idx = np.asarray(g); js = jj_[idx]
        loo_s = sm[inv[js]] - np.where(u[js], v[js], 0); loo_c = ct[inv[js]] - u[js]
        relind[idx] = np.where(loo_c >= 5, x20[dday, js] - loo_s / np.maximum(loo_c, 1), np.nan)
    D['era'] = np.where(dates[dd_] <= pd.Timestamp('2015-12-31'), ERAS[0], ERAS[1])
    D['diff'] = D.rem - D.bench
    feats = {'ΔT_ld10': dT[dd_, jj_], '相對產業20日': relind, '出貨日數20': DIST20[dd_, jj_], '上影線5日': USH5[dd_, jj_],
             '融資增量(成交額天數)': DMR[dd_, jj_], 'EXT12 過強乖離': EXT12[dd_, jj_]}
    flags = {'出池': ~pool[dd_, jj_], '大盤背離': ixhi[dd_] & ~sthi[dd_, jj_], '近5日漲停≥2': LU5[dd_, jj_] >= 2, 'MACD事件近5日': MACD5[dd_, jj_]}
    for k, v in feats.items(): D[k] = v
    for k, v in flags.items(): D[k] = v
    D['tid'] = D.j.astype(np.int64) * 100000 + (D.d - D.held)
    def st(x):
        return dict(diff=x['diff'].mean(), rem=x.rem.mean(), crash=(x.rem <= -0.10).mean(), big=(x.rem >= 0.30).mean(), dead=(x.rem.abs() <= 0.03).mean(), f20=x.f20.mean(), n=len(x), nt=x.tid.nunique())
    def fmt(s):
        return f"{s['diff']:+.1%}｜{s['rem']:+.1%}｜{s['crash']:.0%}｜{s['big']:.0%}｜{s['dead']:.0%}｜{s['f20']:+.1%}（{s['n']}／{s['nt']} 筆）"
    out = [f"# M-195 結果：持有中面板（{'奇' if PARITY else '偶'}數代碼；到 2024-11-13）\n",
           '格式：換股差｜剩餘報酬｜崩跌率（≤−10%）｜大賺率（≥+30%）｜死錢率（|剩餘| ≤ 3%）｜未來 20 日超額（持有日數／交易筆數）。\n',
           f"全體：{fmt(st(D[D.era == ERAS[0]]))} ｜ {fmt(st(D[D.era == ERAS[1]]))}\n", '## 基本狀態\n']
    for v, bins in (('held', [0, 5, 10, 20, 40, 80, 250]), ('unr', [-1, -0.05, 0, 0.05, 0.10, 0.20, 0.40, 10]), ('stopd', [-1, 0.25, 0.5, 0.75, 0.999, 2])):
        q = pd.cut(D[v], bins, include_lowest=True)
        out += [f'### {v}\n', '| 區間 | 2005–15 | 2016–24/11 |', '|---|---|---|']
        for b in q.cat.categories:
            m = q == b
            out.append(f"| {b} | {fmt(st(D[m & (D.era == ERAS[0])]))} | {fmt(st(D[m & (D.era == ERAS[1])]))} |")
        out.append('')
    out += ['## 候選特徵\n', '| 條件 | 2005–15 旗標｜其他 | 2016–24/11 旗標｜其他 | 判定 |', '|---|---|---|---|']
    conds = []
    for k in feats:
        x = D[k]; hi_, lo_ = x.quantile(0.8), x.quantile(0.2)
        conds += [(f'{k} 最高 20%（≥ {hi_:.3g}）', x >= hi_), (f'{k} 最低 20%（≤ {lo_:.3g}）', x <= lo_)]
    conds += [(k, D[k]) for k in flags]
    noms = []
    for nm, f in conds:
        f = f.fillna(False).astype(bool) if hasattr(f, 'fillna') else f
        cells, ok, cr = [], True, True
        for e in ERAS:
            a, b = st(D[f & (D.era == e)]), st(D[~f & (D.era == e)])
            cells.append(f"{fmt(a)}<br>{fmt(b)}")
            ok &= a['n'] >= 200 and a['diff'] <= b['diff'] - 0.03 and a['big'] <= b['big']
            cr &= a['crash'] >= 1.3 * b['crash']
        tag = ('**崩跌型**' if cr else '**死錢型**') if ok else ''
        if ok: noms.append(f'{nm} → {tag}')
        out.append(f"| {nm} | {cells[0]} | {cells[1]} | {tag} |")
    out += ['', '## 提名\n'] + ([f'- {s}' for s in noms] or ['- 無'])
    D.drop(columns=['j']).to_parquet(f'/tmp/M195_D_{PARITY}.parquet')
    return dict(summary='\n'.join(out), headline=(f'提名 {len(noms)} 項') if noms else '無提名', n_variants=0 if PARITY else len(noms))
