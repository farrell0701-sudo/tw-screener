# M-160：營收創高（R4）的超額延續期與每日超額分布（奇數代碼）。定義見 registry_M.md M-160。
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
KS = (-60, -20, -5, 0, 5, 10, 20, 40, 60, 90, 120, 180, 250)
BINS = ((1, 5), (6, 20), (21, 60), (61, 120), (121, 250))
KMAX = 250


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; del S
    U, pool = dbg['U'], dbg['pool']; assert U.shape[1] == N
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    EX = (R - ew[:, None]).astype(np.float32); CX = np.cumsum(EX, axis=0, dtype=np.float64)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); bear = (ix < ix.rolling(200).mean()).shift(1, fill_value=False).to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    W = rev_factors(dates)['R4 營收創高']; jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) - 60 >= 0 and dates.get_loc(d) + KMAX < T]
    karr = np.arange(-60, KMAX + 1)

    # 每期每組：累積路徑（組平均）、每日超額路徑、個股 60／120 日超額
    G = {}   # (layer, grp) -> list of dict(t, path[karr], daily[1..KMAX])
    XS = {}  # (layer, grp) -> list of arrays (t, e60, e120)
    for t in samp:
        m = dates[t].to_period('M') - 1
        if m not in W.index: continue
        v = W.loc[m].to_numpy(float); x = np.where(jm >= 0, v[np.clip(jm, 0, None)], np.nan)
        base = CX[t]
        for layer, mask, ng in (('強勢池', pool[t], 5), ('全市場', U[t], 10)):
            jj = np.nonzero(mask & odd & np.isfinite(x))[0]
            if len(jj) < ng * 5: continue
            g = pd.qcut(pd.Series(x[jj]).rank(method='first').to_numpy(), ng, labels=False)
            groups = {'高': jj[g == ng - 1], '低': jj[g == 0], '整層': jj}
            if layer == '強勢池': groups.update({'創高 R4≥0': jj[x[jj] >= 0], '未創高 R4<0': jj[x[jj] < 0]})
            for gn, sel in groups.items():
                if len(sel) < 3: continue
                path = (CX[t + karr][:, sel] - base[sel]).mean(axis=1)
                G.setdefault((layer, gn), []).append(dict(t=t, path=path))
                XS.setdefault((layer, gn), []).append(np.c_[np.full(len(sel), t), CX[t + 60, sel] - base[sel], CX[t + 120, sel] - base[sel]])
    def era_sel(lst, lo, hi): return [d for d in lst if pd.Timestamp(lo) <= dates[d['t']] <= pd.Timestamp(hi)]
    k0 = 60  # karr 中 k=0 的位置
    L = ['# M-160 結果：營收創高（R4）的超額延續期與每日超額分布（奇數代碼）\n',
         '超額＝扣可交易池等權的累積超額，取樣日 t 收盤為 0；強勢池分五組（高＝Q5、低＝Q1），全市場分十組。\n']
    summ = []
    for layer, (gh, gl) in (('強勢池', ('高', '低')), ('全市場', ('高', '低'))):
        for en, lo, hi in ERAS + (('2005–26', '2005-01-01', '2026-12-31'),):
            H = era_sel(G[(layer, gh)], lo, hi); Lw = era_sel(G[(layer, gl)], lo, hi); A = era_sel(G[(layer, '整層')], lo, hi)
            th = {d['t'] for d in H} & {d['t'] for d in Lw}
            Hm = np.array([d['path'] for d in H if d['t'] in th]); Lm = np.array([d['path'] for d in Lw if d['t'] in th])
            tt = np.array([d['t'] for d in H if d['t'] in th]); Dm = Hm - Lm
            L += [f'## {layer}｜{en}（期數 {len(tt)}）\n', '### 累積超額路徑\n',
                  '| k 日 | ' + ' | '.join(f'{k:+d}' for k in KS) + ' |', '|---' * (len(KS) + 1) + '|']
            idx = [k + k0 for k in KS]
            L.append('| 高組 | ' + ' | '.join(f'{Hm[:, i].mean():+.2%}' for i in idx) + ' |')
            L.append('| 低組 | ' + ' | '.join(f'{Lm[:, i].mean():+.2%}' for i in idx) + ' |')
            L.append('| 整層 | ' + ' | '.join(f'{np.mean([d["path"][i] for d in A]):+.2%}' for i in idx) + ' |')
            L.append('| 高−低 | ' + ' | '.join(f'{Dm[:, i].mean():+.2%}' for i in idx) + ' |')
            L.append('| t | ' + ' | '.join(f'{nw_t(Dm[:, i], max(1, abs(k) // 21)):.1f}' for k, i in zip(KS, idx)) + ' |')
            sp = Dm.mean(axis=0); s250 = sp[-1]
            half = next((k for k in range(1, KMAX + 1) if sp[k + k0] >= s250 / 2), None) if s250 > 0 else None
            inc = [k for k in range(20, KMAX + 1, 10) if sp[k + k0] - sp[k - 20 + k0] > 0]
            last_inc = max(inc) if inc else None
            L.append(f"\n取樣前（−60→0）高−低：{sp[k0] - sp[0]:+.2%}；取樣後（0→+250）：{s250:+.2%}；半程日 {half}；最後一個 20 日視窗仍擴大的結束日 ≈ {last_inc}\n")
            # 每日超額（時間分布）：高−低 每日差
            dd = np.diff(Dm, axis=1)  # dd[:, j] = 第 karr[j+1] 日的每日差
            L += ['### 每日超額（高−低）\n', '| 區段 | 平均每日（基點） | 區段合計 | 正值日比例（期平均後） |', '|---|---|---|---|']
            for a, b in BINS:
                cols = [k + k0 - 1 for k in range(a, b + 1)]
                daily = dd[:, cols].mean(axis=0)
                L.append(f"| {a}–{b} 日 | {daily.mean() * 1e4:+.1f} | {daily.sum():+.2%} | {np.mean(daily > 0):.0%} |")
            L.append('')
            # 個股分布
            if en != '2005–26':
                L += ['### 個股超額分布\n', '| 組 | 期間 | P10 | P25 | 中位數 | P75 | P90 | 平均 | 前 10% 個股占總超額 |', '|---|---|---|---|---|---|---|---|---|']
                for gn in (gh, gl):
                    X = np.vstack(XS[(layer, gn)]); X = X[(dates[X[:, 0].astype(int)] >= pd.Timestamp(lo)) & (dates[X[:, 0].astype(int)] <= pd.Timestamp(hi))]
                    for col, nm in ((1, '60 日'), (2, '120 日')):
                        v = X[:, col]; v = v[np.isfinite(v)]; q = np.percentile(v, [10, 25, 50, 75, 90])
                        top = np.sort(v)[-max(1, len(v) // 10):]; share = top.sum() / v.sum() if v.sum() > 0 else np.nan
                        L.append(f"| {gn} | {nm} | " + ' | '.join(f'{z:+.1%}' for z in q) + f" | {v.mean():+.1%} | {share:.0%} |" if np.isfinite(share) else
                                 f"| {gn} | {nm} | " + ' | '.join(f'{z:+.1%}' for z in q) + f" | {v.mean():+.1%} | （總超額 ≤ 0） |")
                L.append('')
                # 多空頭
                bb = bear[tt]
                L += ['### 多頭／空頭（高−低）\n', '| 狀態 | 期數 | +60 日 | +120 日 | +250 日 |', '|---|---|---|---|---|']
                for nm, mk in (('多頭', ~bb), ('空頭', bb)):
                    if mk.sum(): L.append(f"| {nm} | {mk.sum()} | {Dm[mk, 60 + k0].mean():+.2%} | {Dm[mk, 120 + k0].mean():+.2%} | {Dm[mk, 250 + k0].mean():+.2%} |")
                L.append('')
            summ.append((layer, en, sp[k0] - sp[0], sp[60 + k0], sp[120 + k0], s250, half))
    # 創高事件
    L += ['## 強勢池：創高（R4 ≥ 0）vs 未創高（R4 < 0）累積超額\n', '| 期間 | 組 | 平均家數 | ' + ' | '.join(f'{k:+d}' for k in KS) + ' |', '|---|---|---' + '|---' * len(KS) + '|']
    for en, lo, hi in ERAS:
        for gn in ('創高 R4≥0', '未創高 R4<0'):
            H = era_sel(G[('強勢池', gn)], lo, hi); M_ = np.array([d['path'] for d in H])
            n = np.mean([len(x) for x in XS[('強勢池', gn)]])
            L.append(f"| {en} | {gn} | {n:.0f} | " + ' | '.join(f'{M_[:, k + k0].mean():+.2%}' for k in KS) + ' |')
    head = ['## 摘要（高−低累積超額）\n', '| 層 | 期間 | 取樣前 −60→0 | +60 | +120 | +250 | 半程日 |', '|---|---|---|---|---|---|---|']
    for layer, en, pre, a, b, c, h in summ:
        head.append(f"| {layer} | {en} | {pre:+.2%} | {a:+.2%} | {b:+.2%} | {c:+.2%} | {h} |")
    L[2:2] = head + ['']
    s = [x for x in summ if x[0] == '強勢池' and x[1] == '2005–26'][0]
    return dict(summary='\n'.join(L), headline=f"強勢池 R4 高−低：+60 {s[3]:+.2%}、+120 {s[4]:+.2%}、+250 {s[5]:+.2%}；半程日 {s[6]}", n_variants=0)
