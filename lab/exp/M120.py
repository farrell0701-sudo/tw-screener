# M-120：月營收因子（依 M-119 評估標準；奇數代碼）。定義見 registry_M.md M-120。
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t

TWD = LAB.parents[1] / 'tw-data-v1.1'
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
HS = (5, 20, 60, 120)


def rev_factors(dates=None):
    d = pd.read_parquet(TWD / 'derived' / 'month_revenue.parquet')
    d = d[d.security_key.notna() & d.security_key.ne('')]
    d['m'] = d.period_end.dt.to_period('M')
    W = d.pivot_table(index='m', columns='security_key', values='revenue', aggfunc='last').astype(float)
    W = W.reindex(pd.period_range(W.index.min(), W.index.max(), freq='M')); W[W <= 0] = np.nan
    s3 = W.rolling(3, min_periods=3).sum()
    with np.errstate(all='ignore'):
        R1 = W / W.shift(12) - 1
        R2 = s3 / s3.shift(12) - 1
        R3 = R2 - R2.shift(3)
        R4 = s3 / s3.shift(1).rolling(24, min_periods=12).max() - 1
        mu = R1.shift(1).rolling(12, min_periods=10).mean(); sd = R1.shift(1).rolling(12, min_periods=10).std()
        R5 = (R1 - mu) / sd.where(sd > 0)
    return {'R1 單月年增': R1, 'R2 三月年增': R2, 'R3 年增加速': R3, 'R4 營收創高': R4, 'R5 年增驚喜': R5}


def srank(a):
    return pd.Series(a).rank().to_numpy()


PARITY = 1      # 1＝奇數（探索集）、0＝偶數（確認集）


def evaluate(parity=1, names=None, fac_fn=None, ref=None, eras=None, layers=None, detail=False):
    """fac_fn(dates)→{名稱: 以月份（取樣月−1）為索引的寬表}；ref：同格式的對照因子（報告截面相關）。
    layers(U, pool, dates, keys)→[(層名, T×N 布林遮罩, 組數)]（M-165 起；預設＝全市場十組、強勢池五組）。
    detail=True 時另回傳 {(因子, 層, 期間): 各組統計 DataFrame}。"""
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; del S
    if Tr.shape[1] != N: full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
    R20 = np.asarray(F['R20'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    FAC = (fac_fn or rev_factors)(dates)
    REF = {k: (v, pd.Index(v.columns.astype(str)).get_indexer(keys)) for k, v in (ref or {}).items()}
    if names: FAC = {k: v for k, v in FAC.items() if k in names}
    # 取樣日：每月第一個 日 > 10 的交易日
    ds = pd.Series(dates, index=dates); samp = ds[ds.dt.day > 10].groupby(ds[ds.dt.day > 10].dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) + 120 < T]
    # 預算各取樣日的前瞻量
    fwd = {}
    for t in samp:
        ex = {h: CX[t + h] - CX[t] for h in HS}
        with np.errstate(all='ignore'):
            up = np.nanmax(HA[t + 1:t + 61], axis=0) / Cf[t] - 1; dn = np.nanmin(LA[t + 1:t + 61], axis=0) / Cf[t] - 1; a60 = Cf[t + 60] / Cf[t] - 1
        fwd[t] = (ex, up, dn, a60)
    L = ['# M-120 結果：月營收因子（M-119 標準；奇數代碼；F 計數 5）\n',
         '取樣：每月 10 日後第一個交易日收盤；超額＝扣可交易池等權；最高漲幅／最大跌幅與 60 日絕對報酬皆扣同期同層中位數。\n']
    summ = []; G = {}
    LAYERS = layers(U, pool, dates, keys) if layers else (('全市場', U, 10), ('強勢池', pool, 5))
    for fn, Fm in FAC.items():
        L.append(f'# {fn}\n')
        colix = pd.Index(Fm.columns.astype(str))
        jmap = colix.get_indexer(keys)
        for layer, mask, ng in LAYERS:
            for era, lo, hi in (eras or ERAS):
                ics = {h: [] for h in HS}; rows = []; prev_top = None; keep = []; cov = []; cT = []; cR = []; cF = {k: [] for k in REF}
                for t in samp:
                    if not (pd.Timestamp(lo) <= dates[t] <= pd.Timestamp(hi)): continue
                    m = dates[t].to_period('M') - 1
                    if m not in Fm.index: continue
                    fv = Fm.loc[m].to_numpy(float); x = np.where(jmap >= 0, fv[np.clip(jmap, 0, None)], np.nan)
                    sel = mask[t] & odd; cov.append(np.isfinite(x[sel]).mean() if sel.any() else np.nan)
                    sel = sel & np.isfinite(x)
                    if sel.sum() < ng * 5: continue
                    jj = np.nonzero(sel)[0]; xv = x[jj]
                    ex, up, dn, a60 = fwd[t]
                    for h in HS:
                        y = ex[h][jj]; ok = np.isfinite(y)
                        if ok.sum() > 10: ics[h].append(np.corrcoef(srank(xv[ok]), srank(y[ok]))[0, 1])
                    tv, rv = Tr[t, jj], R20[t, jj]
                    for arr, lst in ((tv, cT), (rv, cR)):
                        ok = np.isfinite(arr)
                        if ok.sum() > 10: lst.append(np.corrcoef(srank(xv[ok]), srank(arr[ok]))[0, 1])
                    for rk, (rf, rj) in REF.items():
                        if m in rf.index:
                            rv2 = rf.loc[m].to_numpy(float); rx = np.where(rj >= 0, rv2[np.clip(rj, 0, None)], np.nan)[jj]; ok = np.isfinite(rx)
                            if ok.sum() > 10: cF[rk].append(np.corrcoef(srank(xv[ok]), srank(rx[ok]))[0, 1])
                    g = pd.qcut(pd.Series(xv).rank(method='first').to_numpy(), ng, labels=False)   # 平手依順序（殖利率等有大量相同值）
                    top = set(jj[g == ng - 1])
                    if prev_top is not None and len(prev_top): keep.append(len(top & prev_top) / len(prev_top))
                    prev_top = top
                    e60 = ex[60][jj]; q10, q90 = np.nanpercentile(e60, [10, 90])
                    mu, md, ma = np.nanmedian(up[jj]), np.nanmedian(dn[jj]), np.nanmedian(a60[jj])
                    for k in range(ng):
                        mm = g == k
                        rows.append(dict(t=t, k=k, e20=np.nanmean(ex[20][jj][mm]), e60=np.nanmean(e60[mm]), a60=np.nanmean(a60[jj][mm]) - ma,
                                         up=np.nanmean(up[jj][mm]) - mu, dn=np.nanmean(dn[jj][mm]) - md,
                                         left=np.nanmean(e60[mm] <= q10), right=np.nanmean(e60[mm] >= q90)))
                if not rows: continue
                D = pd.DataFrame(rows)
                L += [f'## {fn}｜{layer}｜{era}（期數 {D.t.nunique()}，覆蓋 {np.nanmean(cov):.0%}，與 T_ld 相關 {np.mean(cT):+.2f}、與 R20 相關 {np.mean(cR):+.2f}{''.join(f'、與 {k} 相關 {np.mean(v):+.2f}' for k, v in cF.items() if v)}，頭組逐期保留 {np.mean(keep):.0%}）\n',
                      '| 期間 | ' + ' | '.join(f'{h} 日' for h in HS) + ' |', '|---|---|---|---|---|',
                      '| IC 平均（t） | ' + ' | '.join(f"{np.mean(ics[h]):+.3f}（{nw_t(ics[h], max(1, h // 21)):.1f}）" for h in HS) + ' |',
                      '| IC > 0 比例 | ' + ' | '.join(f"{np.mean(np.array(ics[h]) > 0):.0%}" for h in HS) + ' |', '',
                      '| 組 | 後20日超額 | 後60日超額 | 60日絕對（扣中位） | 60日最高漲幅（扣中位） | 60日最大跌幅（扣中位） | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|---|---|']
                for k in range(ng):
                    x = D[D.k == k]
                    L.append(f"| {'低' if k == 0 else ('高' if k == ng - 1 else '')}{k + 1} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {x.a60.mean():+.2%} | {x.up.mean():+.2%} | {x.dn.mean():+.2%} | {x.left.mean():.0%} | {x.right.mean():.0%} |")
                hi_, lo_ = D[D.k == ng - 1].set_index('t'), D[D.k == 0].set_index('t')
                sp20 = (hi_.e20 - lo_.e20).dropna(); sp60 = (hi_.e60 - lo_.e60).dropna()
                L.append(f"| 高−低 | {sp20.mean():+.2%}（t {nw_t(sp20, 1):.1f}） | {sp60.mean():+.2%}（t {nw_t(sp60, 3):.1f}） | | {(hi_.up - lo_.up).mean():+.2%} | {(hi_.dn - lo_.dn).mean():+.2%} | | |\n")
                summ.append((fn, layer, era, [np.mean(ics[h]) for h in HS], sp60.mean(), nw_t(sp60, 3)))
                G[(fn, layer, era)] = D.groupby('k')[['e20', 'e60', 'a60', 'up', 'dn', 'left', 'right']].mean().assign(ic60=np.mean(ics[60]), ic120=np.mean(ics[120]), sp=sp60.mean(), t=nw_t(sp60, 3), n=D.t.nunique())
    return (L, summ, G) if detail else (L, summ)


def main():
    L, summ = evaluate(PARITY)
    head = ['# 摘要（IC 5／20／60／120 日；高−低 60 日超額）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日 |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, _t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%} |")
    return dict(summary='\n'.join(head + [''] + L), headline='月營收因子評估完成', n_variants=0)
