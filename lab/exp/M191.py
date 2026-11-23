# M-191：X1 進場點位分析（位階、狀態、時機；奇數代碼；到 2024-11-13）。定義見 registry_M.md M-191。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M165

END = pd.Timestamp('2024-11-13'); NP = 20
ERAS = ('2005–15', '2016–24/11')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cdf = pd.DataFrame(CA).ffill(); C = Cdf.to_numpy()
    O = np.asarray(F['OA'], np.float64)
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    Rz = np.where(VALID & ~np.isnan(R), R, 0.0)
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, Rz, np.nan), axis=1))
    CX = np.cumsum(Rz - ew[:, None], axis=0)
    ATR = np.asarray(F['ATR'], np.float64)
    EMA12 = Cdf.ewm(span=12, adjust=False).mean().to_numpy(); EMA50 = Cdf.ewm(span=50, adjust=False).mean().to_numpy()
    MX250 = Cdf.rolling(250, min_periods=200).max().to_numpy()
    MX60 = Cdf.rolling(60, min_periods=40).max().to_numpy(); MN60 = Cdf.rolling(60, min_periods=40).min().to_numpy()
    HA, LA = np.asarray(F['HA'], np.float64), np.asarray(F['LA'], np.float64)
    RNG20 = pd.DataFrame(HA).rolling(20, min_periods=15).max().to_numpy() - pd.DataFrame(LA).rolling(20, min_periods=15).min().to_numpy()
    MX20 = Cdf.rolling(20, min_periods=15).max().to_numpy()
    newhi = np.where(np.isfinite(C) & (C >= MX20 - 1e-9), np.arange(T)[:, None], -1)
    lastNH = np.maximum.accumulate(newhi, axis=0)
    lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
    LU5 = pd.DataFrame((np.where(VALID, R, 0) >= 0.95 * lim).astype(float)).rolling(5, min_periods=1).sum().to_numpy()
    PC = np.vstack([np.zeros((1, N)), np.cumsum(pool, axis=0)])
    A = M165.layers(U, pool, dates, keys)[0][1]
    sampT = np.array(sorted(t for _, (t, _) in M165.samp_prev(dates).items()))
    ATRP = ATR / C
    with np.errstate(all='ignore'):
        ar = pd.DataFrame(np.where(U, ATRP, np.nan)).rank(axis=1, pct=True).to_numpy()
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); ixd = (ix / ix.rolling(200).mean() - 1).to_numpy()
    ctx = sim.make_ctx('fix')
    col = {k: i for i, k in enumerate(keys)}; d2i = {d: i for i, d in enumerate(dates)}
    rows = []
    for i in range(NP):
        _, tr, _ = sim.runX(ctx, off=i % 20, seed=1000 + i)
        for code, d0, d1, w, ret, hold, why, st, lk in tr:
            if d0 > END or d1 > END: continue
            j = col[str(code)]
            if not odd[j]: continue
            t0 = d2i[d0]; s = t0 - 1
            past = sampT[sampT <= s][-3:]
            fa = bool(A[past, j].any()) if len(past) else False
            frac = (PC[s - 20, j] - PC[s - 60, j]) / 40.0
            e60 = CX[min(t0 + 59, T - 1), j] - CX[s, j]
            rows.append(dict(
                era=ERAS[0] if d0 <= pd.Timestamp('2015-12-31') else ERAS[1], ret=ret, hold=hold, why=why, e60=e60,
                距250日高=C[s, j] / MX250[s, j], 區間位置60=(C[s, j] - MN60[s, j]) / (MX60[s, j] - MN60[s, j]),
                EXT12=(C[s, j] - EMA12[s, j]) / ATR[s, j], EXT50=(C[s, j] - EMA50[s, j]) / ATR[s, j],
                ATR排名=ar[s, j], ATR擴張=ATR[s, j] / ATR[s - 20, j], 整理緊度=RNG20[s, j] / ATR[s, j],
                池內比例=frac, 來自營收池=fa, 大盤距200MA=ixd[s],
                R5=C[s, j] / C[s - 5, j] - 1, 距20日新高天數=s - lastNH[s, j], 近5日漲停=LU5[s, j], 跳空=O[t0, j] / C[s, j] - 1))
    D = pd.DataFrame(rows)
    D['階段'] = np.where(D.池內比例 <= 0.2, np.where(D.來自營收池, '新進自營收池', '新進非營收'), np.where(D.池內比例 >= 0.8, '老成員', '其他'))
    D['老×高波動'] = (D.階段 == '老成員') & (D.ATR排名 > 0.5)
    # 對照：同日強勢池全體（奇數代碼）在訊號日的位階分布
    ent = D.groupby('era').size() / NP
    def outc(x):
        return f"{x.ret.mean():+.1%}｜{(x.ret > 0).mean():.0%}｜{(x.ret <= -0.10).mean():.0%}｜{(x.ret >= 0.30).mean():.0%}｜{x.hold.mean():.0f}｜{x.e60.mean():+.1%}（{len(x) / NP:.0f}）"
    out = ['# M-191 結果：X1 進場點位（奇數代碼；20 條路徑平均；到 2024-11-13）\n',
           '格式：平均報酬｜勝率｜≤−10%｜≥+30%｜平均持有天數｜60 日超額（每條路徑平均筆數）。左欄 2005–15，右欄 2016–24/11。\n',
           f"整體：{outc(D[D.era == ERAS[0]])} ｜ {outc(D[D.era == ERAS[1]])}\n",
           '出場原因：' + '；'.join(f"{k} {v:.0%}" for k, v in D.why.value_counts(normalize=True).items()) + '\n']
    num = ['距250日高', '區間位置60', 'EXT12', 'EXT50', 'ATR排名', 'ATR擴張', '整理緊度', '大盤距200MA', 'R5', '距20日新高天數', '跳空']
    for v in num:
        x = D[v].replace([np.inf, -np.inf], np.nan)
        try: q = pd.qcut(x, 5, duplicates='drop')
        except ValueError: continue
        out += [f'## {v}（中位數 {x.median():.3g}；10%／90% 分位 {x.quantile(.1):.3g}／{x.quantile(.9):.3g}）\n', '| 五等分 | 2005–15 | 2016–24/11 |', '|---|---|---|']
        for b in q.cat.categories:
            m = q == b
            out.append(f"| {b} | {outc(D[m & (D.era == ERAS[0])])} | {outc(D[m & (D.era == ERAS[1])])} |")
        out.append('')
    for v, lv in (('階段', ['新進自營收池', '新進非營收', '其他', '老成員']), ('近5日漲停', [0, 1, 2, 3]), ('老×高波動', [False, True])):
        out += [f'## {v}\n', '| 組 | 占比 | 2005–15 | 2016–24/11 |', '|---|---|---|---|']
        for g in lv:
            m = (D[v] >= g) if v == '近5日漲停' and g == 3 else (D[v] == g)
            if m.sum() < NP * 5: continue
            out.append(f"| {g}{'+' if v == '近5日漲停' and g == 3 else ''} | {m.mean():.0%} | {outc(D[m & (D.era == ERAS[0])])} | {outc(D[m & (D.era == ERAS[1])])} |")
        out.append('')
    D.to_parquet('/tmp/M191_D.parquet')
    return dict(summary='\n'.join(out), headline=f'X1 進場點位描述（{len(D) / NP:.0f} 筆／路徑）', n_variants=0)
