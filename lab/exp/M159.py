# M-159：池內排序方式比較（營收年增率／T_ld／ΔT_ld10 先後順序；多空頭；奇數代碼）。定義見 registry_M.md M-159。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
from M120 import rev_factors

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
NTOP = 12


def pct(v):
    return pd.Series(v).rank(pct=True, method='average').to_numpy()


def tiered(primary, secondary):
    """主變數十分位（高者先）＋十分位內次變數；回傳合成分數（越大越前）。缺值排最後。"""
    p = pct(np.where(np.isfinite(primary), primary, -np.inf)); d = np.clip(np.ceil(p * 10), 1, 10)
    s = pct(np.where(np.isfinite(secondary), secondary, -np.inf))
    return d + s * 0.999


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; del S
    U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']
    assert Tr.shape[1] == N and U.shape[1] == N
    dT = (pd.DataFrame(Tr) - pd.DataFrame(Tr).shift(10)).to_numpy()
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); bear = (ix < ix.rolling(200).mean()).shift(1, fill_value=False).to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    FAC = rev_factors(dates); R1w, R4w = FAC['R1 單月年增'], FAC['R4 營收創高']
    j1 = pd.Index(R1w.columns.astype(str)).get_indexer(keys); j4 = pd.Index(R4w.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) + 60 < T]

    def getf(W, jm, m):
        if m not in W.index: return np.full(N, np.nan)
        v = W.loc[m].to_numpy(float); return np.where(jm >= 0, v[np.clip(jm, 0, None)], np.nan)

    NAMES = {'S0': 'S0 池內依 T_ld（對照）', 'SX': 'SX 池內依 X1 分數（對照）', 'S1': 'S1 年增率→同階層比 T_ld', 'S2': 'S2 年增率→同階層比 ΔT_ld10',
             'S3': 'S3 池內依 ΔT_ld10', 'S4': 'S4 全市場依 ΔT_ld10', 'S5': 'S5 ΔT_ld10→同階層比年增率',
             "S1'": "S1′ R4→同階層比 T_ld", "S2'": "S2′ R4→同階層比 ΔT_ld10", "S5'": "S5′ ΔT_ld10→同階層比 R4", 'P': 'P 整池平均'}
    rows = []; cov = []
    for t in samp:
        m = dates[t].to_period('M') - 1
        r1, r4 = getf(R1w, j1, m), getf(R4w, j4, m)
        jp = np.nonzero(pool[t] & odd)[0]; ju = np.nonzero(U[t] & odd & np.isfinite(dT[t]))[0]
        if len(jp) < 30: continue
        cov.append(np.isfinite(r1[jp]).mean())
        e20, e60 = CX[t + 20] - CX[t], CX[t + 60] - CX[t]
        q10, q90 = np.nanpercentile(e60[jp], [10, 90])
        tl, dt = Tr[t, jp], dT[t, jp]
        sc = {'S0': pct(tl), 'SX': 0.5 * pct(tl) + 0.5 * pct(np.nan_to_num(dt, nan=-np.inf)),
              'S1': tiered(r1[jp], tl), 'S2': tiered(r1[jp], dt), 'S3': pct(np.nan_to_num(dt, nan=-np.inf)), 'S5': tiered(dt, r1[jp]),
              "S1'": tiered(r4[jp], tl), "S2'": tiered(r4[jp], dt), "S5'": tiered(dt, r4[jp])}
        k20 = max(1, int(round(len(jp) * 0.2)))
        def rec(name, idx, top):
            for tag, sel in (('前12', idx[top[:NTOP]]), ('前20%', idx[top[:k20]])):
                rows.append(dict(t=t, bear=bear[t], s=name, tag=tag, e20=np.nanmean(e20[sel]), e60=np.nanmean(e60[sel]),
                                 win=np.nanmean(e60[sel] > 0), left=np.nanmean(e60[sel] <= q10), right=np.nanmean(e60[sel] >= q90),
                                 r1=np.nanmedian(r1[sel]), tld=np.nanmean(Tr[t, sel] >= np.nanpercentile(Tr[t, jp], 50))))
        for nm, v in sc.items(): rec(nm, jp, np.argsort(-v, kind='stable'))
        rec('S4', ju, np.argsort(-dT[t, ju], kind='stable'))
        for tag in ('前12', '前20%'):
            rows.append(dict(t=t, bear=bear[t], s='P', tag=tag, e20=np.nanmean(e20[jp]), e60=np.nanmean(e60[jp]), win=np.nanmean(e60[jp] > 0),
                             left=0.1, right=0.1, r1=np.nanmedian(r1[jp]), tld=0.5))
    D = pd.DataFrame(rows); D['date'] = dates[D.t]
    L = ['# M-159 結果：池內排序方式比較（奇數代碼；每月取樣）\n',
         f'取樣 {D.t.nunique()} 期（空頭 {D[D.s == "P"].drop_duplicates("t").bear.sum()} 期）；池內 R1 覆蓋 {np.mean(cov):.0%}。超額＝扣可交易池等權。左尾／右尾＝落在同期整池後 60 日超額最差／最好 10%（整池＝10%）。t＝與整池 P 的 60 日差（NW）。\n']
    order = ['S0', 'SX', 'S1', 'S2', 'S3', 'S4', 'S5', "S1'", "S2'", "S5'", 'P']
    for tag in ('前12', '前20%'):
        L.append(f'# 頭部＝{tag}\n')
        for en, lo, hi in ERAS:
            E = D[(D.tag == tag) & (D.date >= lo) & (D.date <= hi)]
            base = E[E.s == 'P'].set_index('t').e60
            L += [f'## {en}\n', '| 排序方式 | 全部 20 日 | 全部 60 日（t） | 勝率 | 左尾 | 右尾 | 多頭 60 日 | 空頭 60 日 | 空頭左尾 | 頭部 R1 中位 |', '|---|---|---|---|---|---|---|---|---|---|']
            for s in order:
                x = E[E.s == s]
                if not len(x): continue
                d = (x.set_index('t').e60 - base).dropna(); bu, be = x[~x.bear], x[x.bear]
                tt = '' if s == 'P' else f'（{nw_t(d, 3):.1f}）'
                L.append(f"| {NAMES[s]} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%}{tt} | {x.win.mean():.0%} | {x.left.mean():.0%} | {x.right.mean():.0%} | {bu.e60.mean():+.2%} | {be.e60.mean():+.2%} | {be.left.mean():.0%} | {x.r1.median():+.0%} |")
            L.append('')
    # 摘要：前 12、兩期 × 多空
    L.insert(2, '## 摘要：前 12 名後 60 日超額（2005–15 多頭／空頭｜2016–26 多頭／空頭）\n')
    summ = ['| 排序方式 | 05–15 多頭 | 05–15 空頭 | 16–26 多頭 | 16–26 空頭 | 全期 |', '|---|---|---|---|---|---|']
    for s in order:
        cells = []
        for en, lo, hi in ERAS:
            x = D[(D.tag == '前12') & (D.s == s) & (D.date >= lo) & (D.date <= hi)]
            cells += [f'{x[~x.bear].e60.mean():+.2%}', f'{x[x.bear].e60.mean():+.2%}']
        a = D[(D.tag == '前12') & (D.s == s)]
        summ.append(f"| {NAMES[s]} | " + ' | '.join(cells) + f" | {a.e60.mean():+.2%} |")
    L[3:3] = summ + ['']
    best = D[(D.tag == '前12') & ~D.s.isin(['P'])].groupby('s').e60.mean().sort_values(ascending=False)
    head = '前12名60日超額最高：' + '、'.join(f"{s} {v:+.2%}" for s, v in best.head(3).items())
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
