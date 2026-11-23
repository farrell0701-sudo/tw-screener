# M-183：第 1 層｜股池的基本特性（奇數代碼；研究期間到 2024-11-13）。定義見 registry_M.md M-183。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M165

END = pd.Timestamp('2024-11-13')
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–24/11', '2016-01-01', '2024-11-13'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy(); ind = meta['industry'].astype(str).to_numpy()
    odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan); LA = np.asarray(F['LA'], np.float64)
    ATRP = np.asarray(F['ATR'], np.float64) / np.asarray(F['CA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy(); Craw = pd.DataFrame(np.where(np.asarray(P['C']) > 0, P['C'], np.nan)).ffill(limit=5).to_numpy()
    AMT = np.asarray(P['AMT'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); bull = (ix > ix.rolling(200).mean()).shift(1).fillna(False).to_numpy()
    A = M165.layers(U, pool, dates, keys)[0][1]
    SH = M165.shares_table(dates); shj = pd.Index(SH.columns.astype(str)).get_indexer(keys)
    rows, struct, flow = [], [], []
    sp = sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0])
    prev = None
    for i, (m, (t, tp)) in enumerate(sp):
        if dates[t] < pd.Timestamp('2005-01-01') or dates[t] > END or t + 121 >= T: prev = None; continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–24/11'
        Ut = U[t]; strong = pool[t] & Ut; rev = A[t] & Ut; rest = Ut & ~strong & ~rev
        sh = np.where(shj >= 0, SH.loc[m].to_numpy(float)[np.clip(shj, 0, None)], np.nan) if m in SH.index else np.full(N, np.nan)
        mc = Craw[tp] * sh
        amt20 = np.nanmean(AMT[max(0, tp - 19):tp + 1], axis=0)
        pr = lambda x: pd.Series(np.where(Ut, x, np.nan)).rank(pct=True).to_numpy()
        mcp, amp = pr(mc), pr(amt20)
        e20, e60, e120 = CX[t + 20] - CX[t], CX[t + 60] - CX[t], CX[t + 120] - CX[t]
        a60 = Cf[t + 60] / Cf[t] - 1
        with np.errstate(all='ignore'): mdd = np.nanmin(LA[t + 1:t + 61], axis=0) / Cf[t] - 1
        q10, q90 = np.nanpercentile(e60[Ut], [10, 90])
        cur = {}
        for nm, msk in (('U 可交易池', Ut), ('強勢池', strong), ('營收池', rev), ('其餘', rest)):
            s = msk & odd; jj = np.nonzero(s)[0]; cur[nm] = set(np.nonzero(msk)[0])
            if not len(jj): continue
            iv = pd.Series(ind[jj]).value_counts(normalize=True)
            struct.append((era, nm, len(jj), np.nanmedian(mcp[jj]), np.nanmedian(amp[jj]), np.nanmedian(ATRP[tp, jj]), iv.head(3).sum(), '、'.join(iv.head(3).index)))
            for j in jj: rows.append((era, nm, bool(bull[t]), e20[j], e60[j], e120[j], a60[j], mdd[j], q10, q90))
        if prev is not None:
            for nm in ('強勢池', '營收池', 'U 可交易池'):
                if prev[nm]: flow.append((era, nm, len(prev[nm] & cur[nm]) / len(prev[nm]),
                                          len(prev['強勢池'] & cur['營收池']) / max(1, len(prev['強勢池'])), len(prev['營收池'] & cur['強勢池']) / max(1, len(prev['營收池']))))
        prev = cur
    D = pd.DataFrame(rows, columns=['era', 'pool', 'bull', 'e20', 'e60', 'e120', 'a60', 'mdd', 'q10', 'q90'])
    St = pd.DataFrame(struct, columns=['era', 'pool', 'n', 'mcp', 'amp', 'atrp', 'top3', 'ind3'])
    Fl = pd.DataFrame(flow, columns=['era', 'pool', 'keep', 's2r', 'r2s'])
    order = ['U 可交易池', '強勢池', '營收池', '其餘']
    L = ['# M-183 結果：第 1 層｜股池的基本特性（奇數代碼；2005–15、2016–2024/11）\n', '## 1. 規模與結構（每月平均）\n',
         '| 股池 | 期間 | 檔數（奇數） | 市值百分位中位數 | 成交金額百分位中位數 | ATR% 中位數 | 前三大產業占比 | 最常見產業 | 下月保留率 |', '|---|---|---|---|---|---|---|---|---|']
    for era in ('2005–15', '2016–24/11'):
        for nm in order:
            x = St[(St.era == era) & (St.pool == nm)]
            if not len(x): continue
            kp = Fl[(Fl.era == era) & (Fl.pool == nm)].keep.mean() if nm in ('強勢池', '營收池', 'U 可交易池') else np.nan
            L.append(f"| {nm} | {era} | {x.n.mean():.0f} | {x.mcp.mean():.0%} | {x.amp.mean():.0%} | {x.atrp.mean():.2%} | {x.top3.mean():.0%} | {x.ind3.mode().iloc[0]} | {'—' if np.isnan(kp) else f'{kp:.0%}'} |")
    L += ['', '## 2. 報酬分布（扣 U 等權為超額；左右尾以同期 U 的 60 日超額 10%／90% 門檻）\n',
          '| 股池 | 期間 | 20 日 | 60 日 | 120 日 | 60 日中位數 | 60 日標準差 | 60 日偏態 | 勝率 | 左尾 | 右尾 | 60 日最大跌幅中位數 | 60 日絕對報酬 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for era in ('2005–15', '2016–24/11'):
        for nm in order:
            x = D[(D.era == era) & (D.pool == nm)]
            L.append(f"| {nm} | {era} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {x.e120.mean():+.2%} | {x.e60.median():+.2%} | {x.e60.std():.1%} | {x.e60.skew():+.2f} | {(x.e60 > 0).mean():.0%} | "
                     f"{(x.e60 <= x.q10).mean():.0%} | {(x.e60 >= x.q90).mean():.0%} | {x.mdd.median():+.1%} | {x.a60.mean():+.2%} |")
    L += ['', '## 3. 多空時期（取樣日前一日加權指數 vs 200 日均線）\n', '| 股池 | 期間 | 多頭 60 日超額 | 空頭 60 日超額 | 多頭 60 日絕對 | 空頭 60 日絕對 | 空頭樣本占比 |', '|---|---|---|---|---|---|---|']
    for era in ('2005–15', '2016–24/11'):
        for nm in order:
            x = D[(D.era == era) & (D.pool == nm)]; b, s = x[x.bull], x[~x.bull]
            L.append(f"| {nm} | {era} | {b.e60.mean():+.2%} | {s.e60.mean():+.2%} | {b.a60.mean():+.2%} | {s.a60.mean():+.2%} | {len(s) / len(x):.0%} |")
    L += ['', '## 4. 強勢池與營收池的流動（下月）\n', '| 期間 | 強勢池成員下月落入營收池 | 營收池成員下月進入強勢池 |', '|---|---|---|']
    for era in ('2005–15', '2016–24/11'):
        x = Fl[(Fl.era == era) & (Fl.pool == '強勢池')]
        L.append(f"| {era} | {x.s2r.mean():.1%} | {x.r2s.mean():.1%} |")
    return dict(summary='\n'.join(L), headline='第 1 層股池基本特性完成', n_variants=0)
