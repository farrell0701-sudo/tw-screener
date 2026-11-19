# M-108：大盤空頭與反彈時期的策略表現拆解（以時期為單位；描述性）。定義見 registry_M.md M-108。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals, sim

TH = 0.15; REB = 60; START = pd.Timestamp('2005-01-01')


def episodes(ix):
    """前高回落 ≥ TH 的空頭：（高點, 低點）。低點＝收復前高前的最低點（或資料結束）。"""
    ix = ix.dropna(); eps = []; i = 0; v = ix.values; idx = ix.index
    peak_i = 0
    while i < len(v):
        if v[i] >= v[peak_i]: peak_i = i
        if v[i] / v[peak_i] - 1 <= -TH:
            j = i
            while j < len(v) and v[j] < v[peak_i]: j += 1
            seg = v[peak_i:j]; tr = peak_i + int(np.argmin(seg))
            eps.append((idx[peak_i], idx[tr], float(v[tr] / v[peak_i] - 1)))
            i = j; peak_i = min(j, len(v) - 1); continue
        i += 1
    return eps


def stats(r, x, e=None):
    if len(r) < 2: return {}
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    c = np.cov(r.values, x.values); beta = float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else np.nan
    out = dict(ret=float(eq.iloc[-1] - 1), dd=dd, beta=beta)
    if e is not None: out['expo'] = float(e.mean())
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    R = sim.run_paths('fix', 100, sig='fix')
    pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    rC = pm([sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(100)]); rX = pm([R['X'][i][0] for i in range(100)]); rV = pm([R['V'][i][0] for i in range(100)])
    eX = pm([R['X'][i][2] for i in range(100)]); eV = pm([R['V'][i][2] for i in range(100)])
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); xr = ix.pct_change()
    idx = rC.loc[START:].index
    rC, rX, rV, eX, eV, xr, ixs = rC.loc[idx], rX.loc[idx], rV.loc[idx], eX.reindex(idx), eV.reindex(idx), xr.loc[idx], ix.loc[idx]
    vol = rC.rolling(60).std() * np.sqrt(245); vmed = float(vol.median())
    EP = episodes(ixs)
    phase = pd.Series('其他', index=idx)
    rows = []
    for k, (pk, tr, depth) in enumerate(EP, 1):
        a = idx.get_loc(pk); b = idx.get_loc(tr)
        segs = [('下跌段', idx[a + 1:b + 1]), ('反彈段', idx[b + 1:b + 1 + REB])]
        for nm, s in segs:
            if len(s) < 2: continue
            phase.loc[s] = nm
            d = dict(次=k, 高點=str(pk.date()), 低點=str(tr.date()), 跌幅=depth, 段=nm, 天數=len(s), 指數=float((1 + xr.loc[s]).prod() - 1),
                     起始波動倍數=float(vol.loc[s[0]] / vmed) if np.isfinite(vol.loc[s[0]]) else np.nan)
            for tag, r, e in (('合併', rC, None), ('X1', rX, eX), ('V4.1', rV, eV)):
                st = stats(r.loc[s], xr.loc[s], None if e is None else e.loc[s])
                for kk, vv in st.items(): d[f'{tag}{kk}'] = vv
            rows.append(d)
    T = pd.DataFrame(rows)
    # 彙總：各段占全期的報酬（對數）與天數
    lr = np.log1p(rC); tot = lr.sum()
    agg = []
    for nm in ('下跌段', '反彈段', '其他'):
        m = phase == nm
        agg.append(dict(段=nm, 天數比例=float(m.mean()), 合併對數報酬占比=float(lr[m].sum() / tot), 合併年化=float(np.expm1(lr[m].mean() * 245)),
                        X1年化=float(np.expm1(np.log1p(rX[m]).mean() * 245)), V年化=float(np.expm1(np.log1p(rV[m]).mean() * 245)),
                        指數年化=float(np.expm1(np.log1p(xr[m].fillna(0)).mean() * 245)),
                        合併beta=float(np.cov(rC[m], xr[m].fillna(0))[0, 1] / np.var(xr[m].fillna(0), ddof=1)),
                        X1曝險=float(eX[m].mean()), V曝險=float(eV[m].mean())))
    A = pd.DataFrame(agg)
    # 全期最大回撤落在哪些段
    eq = (1 + rC).cumprod(); ddser = eq / eq.cummax() - 1
    L = ['# M-108 結果：大盤空頭與反彈時期的策略表現（新基準 100 路徑平均日報酬；指數為價格指數）\n',
         f"空頭定義：加權指數從前高回落 ≥ {TH:.0%}；反彈段＝低點後 {REB} 個交易日。共 {len(EP)} 次。策略近 60 日年化波動全期中位數 {vmed:.1%}。\n",
         '## 各次空頭\n',
         '| 次 | 高點 → 低點 | 跌幅 | 段 | 天數 | 指數 | 合併 | X1 | V4.1 | 合併 beta | X1 曝險 | V4.1 曝險 | 合併段內回撤 | 起始波動倍數 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in T.iterrows():
        L.append(f"| {r.次} | {r.高點} → {r.低點} | {r.跌幅:.0%} | {r.段} | {r.天數} | {r.指數:+.1%} | {r.合併ret:+.1%} | {r.X1ret:+.1%} | {r['V4.1ret']:+.1%} | "
                 f"{r.合併beta:.2f} | {r.X1expo:.2f} | {r['V4.1expo']:.2f} | {r.合併dd:.1%} | {r.起始波動倍數:.1f} |")
    L += ['\n## 彙總（各段合計，年化以該段日報酬平均換算）\n',
          '| 段 | 天數比例 | 合併報酬占全期（對數） | 合併年化 | X1 年化 | V4.1 年化 | 指數年化 | 合併 beta | X1 曝險 | V4.1 曝險 |', '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in A.iterrows():
        L.append(f"| {r.段} | {r.天數比例:.0%} | {r.合併對數報酬占比:+.0%} | {r.合併年化:+.1%} | {r.X1年化:+.1%} | {r.V年化:+.1%} | {r.指數年化:+.1%} | {r.合併beta:.2f} | {r.X1曝險:.2f} | {r.V曝險:.2f} |")
    # 前 5 大回撤區間屬於哪一段
    L.append('\n## 合併版前 5 大回撤區間與所屬段\n')
    used = pd.Series(False, index=idx); k = 0
    while k < 5:
        c = ddser[~used]
        if c.empty or c.min() >= 0: break
        trd = c.idxmin(); pkd = eq.loc[:trd].idxmax(); nxt = eq.loc[trd:]; rec = nxt[nxt >= eq.loc[pkd]]
        end = rec.index[0] if len(rec) else idx[-1]; used.loc[pkd:end] = True
        seg = phase.loc[pkd:trd].value_counts(normalize=True)
        L.append(f"- {pkd.date()} → {trd.date()}：{float(ddser.loc[trd]):.1%}；期間指數 {float(ixs.loc[trd] / ixs.loc[pkd] - 1):+.1%}；天數分布 " + '、'.join(f'{a} {b:.0%}' for a, b in seg.items()))
        k += 1
    return dict(summary='\n'.join(L), headline=f'{len(EP)} 次空頭的時期拆解完成', n_variants=0)
