# M-158：月營收資訊是否已被 T_ld 前 10% 吸收（奇數代碼）。定義與判讀標準見 registry_M.md M-158。
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
TIERS = ('低', '中', '高')


def prank(x, mask):
    out = np.full(x.shape, np.nan); jj = np.nonzero(mask & np.isfinite(x))[0]
    if len(jj): out[jj] = pd.Series(x[jj]).rank(pct=True).to_numpy()
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; del S
    U, pool, hot, steep, Tr = dbg['U'], dbg['pool'], dbg['hot'], dbg['steep'], dbg['Tr']
    if Tr.shape[1] != N:
        full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
        for nm in ('U', 'pool', 'hot', 'steep'):
            a = dbg[nm]
            if a.shape[1] != N: b = np.zeros((T, N), bool); b[:, dbg['cols']] = a; dbg[nm] = b
        U, pool, hot, steep = dbg['U'], dbg['pool'], dbg['hot'], dbg['steep']
    dT = (pd.DataFrame(Tr) - pd.DataFrame(Tr).shift(10)).to_numpy()
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); LA = np.asarray(F['LA'], np.float64)
    VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    FAC = rev_factors(dates); FAC = {'R4 營收創高': FAC['R4 營收創高'], 'R5 年增驚喜': FAC['R5 年增驚喜']}
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) + 120 < T]

    L = ['# M-158 結果：月營收資訊是否已被 T_ld 前 10% 吸收（奇數代碼）\n']
    verdict = {}
    for fn, Fm in FAC.items():
        jmap = pd.Index(Fm.columns.astype(str)).get_indexer(keys)
        A, B, X12, C = [], [], [], []
        for t in samp:
            m = dates[t].to_period('M') - 1
            if m not in Fm.index: continue
            fv = Fm.loc[m].to_numpy(float); x = np.where(jmap >= 0, fv[np.clip(jmap, 0, None)], np.nan)
            mU = U[t] & odd & np.isfinite(x); mP = pool[t] & odd & np.isfinite(x)
            if mU.sum() < 100 or mP.sum() < 30: continue
            e20, e60 = CX[t + 20] - CX[t], CX[t + 60] - CX[t]
            with np.errstate(all='ignore'): dn = np.nanmin(LA[t + 1:t + 61], axis=0) / Cf[t] - 1
            # (A) 富集度
            ru = prank(x, mU); qU = np.clip(np.ceil(np.nan_to_num(ru) * 5) - 1, 0, 4)
            jp = np.nonzero(mP)[0]
            A.append(dict(t=t, **{f'q{k}': np.mean(qU[jp] == k) for k in range(5)}, pct=np.nanmean(ru[jp])))
            # 池內同期尾端門檻
            q10, q90 = np.nanpercentile(e60[jp], [10, 90]); mdn = np.nanmedian(dn[jp])
            def split(mask_tier, layer, tier):
                jj = np.nonzero(mask_tier)[0]
                if len(jj) < 6: return
                xm = np.median(x[jj]); hi, lo = jj[x[jj] > xm], jj[x[jj] <= xm]
                if len(hi) < 2 or len(lo) < 2: return
                B.append(dict(t=t, layer=layer, tier=tier, n=len(jj), d20=np.nanmean(e20[hi]) - np.nanmean(e20[lo]),
                              d60=np.nanmean(e60[hi]) - np.nanmean(e60[lo]),
                              hl=np.mean(e60[hi] <= q10), ll=np.mean(e60[lo] <= q10), hr=np.mean(e60[hi] >= q90), lr=np.mean(e60[lo] >= q90),
                              hdn=np.nanmean(dn[hi]) - mdn, ldn=np.nanmean(dn[lo]) - mdn))
            # (B1) T_ld 池內三檔
            rt = prank(Tr[t], pool[t])                       # 池內百分位以整池計
            for k, tn in enumerate(TIERS):
                split(mP & (rt > k / 3) & (rt <= (k + 1) / 3 + 1e-12), 'B1 T_ld', tn)
            # (B2) X1 候選、X1 分數三檔
            cand = pool[t] & ~hot[t] & ~steep[t]
            scx = 0.5 * np.nan_to_num(prank(Tr[t], pool[t])) + 0.5 * np.nan_to_num(prank(dT[t], pool[t]))
            mC = cand & odd & np.isfinite(x); rc = prank(scx, mC)
            for k, tn in enumerate(TIERS):
                split(mC & (rc > k / 3) & (rc <= (k + 1) / 3 + 1e-12), 'B2 X1 分數', tn)
            jc = np.nonzero(mC)[0]
            if len(jc) >= 12:
                top = jc[np.argsort(-scx[jc], kind='stable')[:12]]
                xm = np.median(x[np.nonzero(mP)[0]])        # 以池內 R4 中位數切
                X12 += [dict(t=t, m=str(m), hi=bool(x[j] > xm), e60=e60[j]) for j in top]
            # (C) 追上
            rT0 = prank(Tr[t], U[t]); rT1 = prank(Tr[t + 60], U[t + 60]); xm = np.median(x[jp])
            for j in jp:
                C.append(dict(t=t, hi=bool(x[j] > xm), d=rT1[j] - rT0[j] if np.isfinite(rT1[j]) else np.nan, stay=bool(pool[t + 60, j])))
        A, B, X12, C = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(X12), pd.DataFrame(C)
        def era(D, lo, hi): return D[(dates[D.t] >= pd.Timestamp(lo)) & (dates[D.t] <= pd.Timestamp(hi))]
        L += [f'# {fn}\n', f'## (A) 富集度：前 10% 池內股票落在全市場 {fn[:2]} 各五分組的比例（均勻＝20%）\n',
              '| 期間 | 期數 | Q1 最低 | Q2 | Q3 | Q4 | Q5 最高 | 池內平均全市場百分位 |', '|---|---|---|---|---|---|---|---|']
        a_ok = []
        for en, lo, hi in ERAS:
            D = era(A, lo, hi); a_ok.append(D.q4.mean() >= 0.40)
            L.append(f"| {en} | {len(D)} | " + ' | '.join(f'{D[f"q{k}"].mean():.0%}' for k in range(5)) + f" | {D.pct.mean():.0%} |")
        L.append('')
        res = {}
        for layer in ('B1 T_ld', 'B2 X1 分數'):
            L += [f'## ({layer[:2]}) {layer[3:]}三檔：檔內 {fn[:2]} 上半 − 下半\n',
                  '| 期間 | 檔 | 平均檔內 n | 20 日差 | 60 日差（NW t） | 左尾 上／下 | 右尾 上／下 | 60 日最大跌幅 上／下（扣池中位） |', '|---|---|---|---|---|---|---|---|']
            for en, lo, hi in ERAS + (('2005–26', '2005-01-01', '2026-12-31'),):
                for tn in TIERS:
                    D = era(B[(B.layer == layer) & (B.tier == tn)], lo, hi)
                    if not len(D): continue
                    res[(layer, en, tn)] = (D.d60.mean(), nw_t(D.d60.dropna(), 3))
                    L.append(f"| {en} | {tn} | {D.n.mean():.0f} | {D.d20.mean():+.2%} | {D.d60.mean():+.2%}（{nw_t(D.d60.dropna(), 3):.1f}） | {D.hl.mean():.0%}／{D.ll.mean():.0%} | {D.hr.mean():.0%}／{D.lr.mean():.0%} | {D.hdn.mean():+.1%}／{D.ldn.mean():+.1%} |")
            L.append('')
        L += [f'## (B2) X1 分數前 12 名（奇數代碼）：池內 {fn[:2]} 中位數以上 vs 以下，後 60 日超額（月份群集 bootstrap）\n',
              '| 期間 | 上半 n | 下半 n | 上半平均 | 下半平均 | 差 | P(差 > 0) |', '|---|---|---|---|---|---|---|']
        rng = np.random.default_rng(20261121)
        for en, lo, hi in ERAS + (('2005–26', '2005-01-01', '2026-12-31'),):
            D = era(X12, lo, hi); g = {mm: x for mm, x in D.groupby('m')}; ms = list(g); cnt = 0
            for _ in range(2000):
                pick = pd.concat([g[ms[k]] for k in rng.choice(len(ms), len(ms))])
                cnt += pick[pick.hi].e60.mean() - pick[~pick.hi].e60.mean() > 0
            h, l = D[D.hi].e60, D[~D.hi].e60
            L.append(f"| {en} | {len(h)} | {len(l)} | {h.mean():+.2%} | {l.mean():+.2%} | {h.mean() - l.mean():+.2%} | {cnt / 2000:.2f} |")
        L += [f'\n## (C) 追上：池內 {fn[:2]} 上半 vs 下半，60 日後\n', '| 期間 | 全市場 T_ld 百分位變化 上／下 | 仍在前 10% 上／下 |', '|---|---|---|']
        for en, lo, hi in ERAS:
            D = era(C, lo, hi)
            L.append(f"| {en} | {D[D.hi].d.mean():+.3f}／{D[~D.hi].d.mean():+.3f} | {D[D.hi].stay.mean():.0%}／{D[~D.hi].stay.mean():.0%} |")
        L.append('')
        # 判讀（R4 才判定）
        if fn.startswith('R4'):
            c1 = all(a_ok)
            def hi_ok(layer):
                return all(res[(layer, e, '高')][0] > 0 for e, _, _ in ERAS) and res[(layer, '2005–26', '高')][1] >= 2
            def weak(layer):
                return all(res[(layer, e, '高')][0] < 0.5 * res[(layer, e, '低')][0] for e, _, _ in ERAS)
            c2 = hi_ok('B1 T_ld') and hi_ok('B2 X1 分數')
            c3 = {ly: weak(ly) for ly in ('B1 T_ld', 'B2 X1 分數')}
            verdict = dict(c1=c1, c2=c2, c2a=hi_ok('B1 T_ld'), c2b=hi_ok('B2 X1 分數'), c3=c3)
            L += ['## 判讀（事前固定）\n',
                  f"1. 進池已大量吸收（最高組比例兩期 ≥ 40%）：{'成立' if c1 else '不成立'}",
                  f"2. 最高檔仍有資訊（B1、B2 高檔兩期 > 0 且合併 t ≥ 2）：{'成立' if c2 else '不成立'}（B1 {'✓' if verdict['c2a'] else '✗'}、B2 {'✓' if verdict['c2b'] else '✗'}）",
                  f"3. 高檔明顯變弱（< 低檔一半，兩期）：B1 {'是' if c3['B1 T_ld'] else '否'}、B2 {'是' if c3['B2 X1 分數'] else '否'}\n"]
    head = (f"R4：進池大量吸收{'成立' if verdict.get('c1') else '不成立'}；最高檔仍有資訊{'成立' if verdict.get('c2') else '不成立'}"
            f"（B1 {'✓' if verdict.get('c2a') else '✗'}、B2 {'✓' if verdict.get('c2b') else '✗'}）")
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
