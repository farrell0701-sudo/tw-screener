# M-164：池外營收創高的超額與出場方式比較（奇數代碼）。定義見 registry_M.md M-164。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
from M120 import rev_factors

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
KS = (-60, -20, 0, 20, 60, 120, 180, 250)
COST = 0.0077; SLIP = 0.0015; NTOP = 10; KEEP = 20; HMAX = 250
EXITS = ('E1 固定 60 日', 'E2 固定 120 日', 'E3 固定 180 日', 'E4 120 日＋15% 硬停損', 'E5 15% 硬停損（≤250 日）',
         'E6 15% 移動停損（≤250 日）', 'E7 20% 移動停損（≤250 日）', 'E8 每月重選＋15% 硬停損')
SIMPLE = {e: k for e, k in zip(EXITS, (0, 0, 0, 1, 1, 2, 2, 3))}


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; del S
    assert U.shape[1] == N
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    AMT20 = np.asarray(F['AMT20'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0); LEW = np.cumsum(np.log1p(ew))
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    W = rev_factors(dates)['R4 營收創高']; jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp_all = d10.groupby(d10.dt.to_period('M')).min()
    samp_all = np.array([dates.get_loc(d) for d in samp_all if d >= pd.Timestamp('2004-06-01')])
    samp = [t for t in samp_all if dates[t] >= pd.Timestamp('2005-01-01') and t + HMAX + 1 < T]
    karr = np.arange(-60, 251)

    def r4(t):
        m = dates[t].to_period('M') - 1
        if m not in W.index: return np.full(N, np.nan)
        v = W.loc[m].to_numpy(float); return np.where(jm >= 0, v[np.clip(jm, 0, None)], np.nan)
    ranks = {}   # 每個取樣日：池外奇數代碼 R4 排名（0＝最高）
    for t in samp_all:
        x = r4(t); jj = np.nonzero(U[t] & ~pool[t] & odd & np.isfinite(x))[0]
        if len(jj) < 50: continue
        order = jj[np.argsort(-x[jj], kind='stable')]; ranks[t] = (x, jj, order)

    # A 部分
    A = []; XS = []
    for t in samp:
        if t not in ranks: continue
        x, jj, order = ranks[t]
        g = pd.qcut(pd.Series(x[jj]).rank(method='first').to_numpy(), 10, labels=False)
        for gn, sel in (('最高組', jj[g == 9]), ('最低組', jj[g == 0]), ('前 10 名', order[:NTOP])):
            A.append(dict(t=t, g=gn, path=(CX[t + karr][:, sel] - CX[t, sel]).mean(axis=1),
                          amt=np.nanmedian(AMT20[t, sel])))
        top = order[:NTOP]; XS.append(np.c_[np.full(len(top), t), CX[t + 60, top] - CX[t, top], CX[t + 120, top] - CX[t, top]])
    A = pd.DataFrame(A); A['date'] = dates[A.t]; XS = np.vstack(XS)
    amt_pool = np.nanmedian(np.where(pool, AMT20, np.nan)[np.array(samp)], axis=1).mean()

    # B 部分：出場模擬
    samp_set = sorted(ranks)
    def trade(t, j):
        if not (VALID[t + 1, j] and np.isfinite(OA[t + 1, j]) and OA[t + 1, j] > 0): return None
        px = OA[t + 1, j] * (1 + SLIP); idx = np.arange(t + 1, min(t + 1 + HMAX, T))
        v = VALID[idx, j]; c = CA[idx, j]
        lastv = np.nonzero(v & np.isfinite(c))[0]
        if len(lastv) == 0: return None
        end_valid = lastv[-1]                                      # 停牌／下市：以最後有效收盤出場
        cc = pd.Series(np.where(v, c, np.nan)).ffill().to_numpy()
        hi = np.maximum.accumulate(np.where(np.isfinite(cc), cc, px)); hi = np.maximum(hi, px)
        res = {}
        def out(k, why_px=None):
            k = min(k, end_valid); p = cc[k] if why_px is None else why_px
            ret = p / px - 1 - COST + SLIP; hold = k + 1
            ex = ret - (np.exp(LEW[idx[k]] - LEW[t]) - 1)
            return (ret, hold, ex)
        for e, n in (('E1 固定 60 日', 60), ('E2 固定 120 日', 120), ('E3 固定 180 日', 180)):
            res[e] = out(n - 1)
        hs = np.nonzero(cc < px * 0.85)[0]; h0 = hs[0] if len(hs) else 10 ** 9
        res['E4 120 日＋15% 硬停損'] = out(min(h0, 119)); res['E5 15% 硬停損（≤250 日）'] = out(min(h0, len(idx) - 1))
        for e, w in (('E6 15% 移動停損（≤250 日）', 0.15), ('E7 20% 移動停損（≤250 日）', 0.20)):
            ts = np.nonzero(cc < hi * (1 - w))[0]; res[e] = out(min(ts[0] if len(ts) else 10 ** 9, len(idx) - 1))
        # E8：後續取樣日不在前 KEEP 名 → 隔日開盤出場
        k8 = len(idx) - 1
        for s in samp_set:
            if s <= t: continue
            if s >= idx[-1]: break
            if j not in set(ranks[s][2][:KEEP]):
                k8 = s + 1 - (t + 1); break
        if h0 <= k8: res['E8 每月重選＋15% 硬停損'] = out(h0)
        else:
            kk = min(k8, end_valid); po = OA[idx[kk], j] * (1 - SLIP) if np.isfinite(OA[idx[kk], j]) and VALID[idx[kk], j] else cc[kk]
            res['E8 每月重選＋15% 硬停損'] = out(kk, po)
        return res
    rows = []
    for t in samp:
        if t not in ranks: continue
        for j in ranks[t][2][:NTOP]:
            r = trade(t, j)
            if r is None: continue
            for e, (ret, hold, ex) in r.items(): rows.append((t, e, ret, hold, ex))
    B = pd.DataFrame(rows, columns=['t', 'e', 'ret', 'hold', 'ex']); B['date'] = dates[B.t]

    L = ['# M-164 結果：池外營收創高的超額與出場方式比較（奇數代碼）\n',
         '池外＝可交易池中排除當日 T_ld 前 10%；每月 10 日後第一個交易日依上月營收 R4 排名。超額＝扣可交易池等權。\n',
         '## A 部分：累積超額路徑\n']
    k0 = 60
    for en, lo, hi in ERAS:
        E = A[(A.date >= lo) & (A.date <= hi)]
        L += [f'### {en}（期數 {E.t.nunique()}）\n', '| 組 | ' + ' | '.join(f'{k:+d}' for k in KS) + ' | 日成交金額中位數（億） |', '|---' * (len(KS) + 2) + '|']
        M = {g: np.array(list(E[E.g == g].path)) for g in ('最高組', '最低組', '前 10 名')}
        for g in M:
            L.append(f"| {g} | " + ' | '.join(f'{M[g][:, k + k0].mean():+.2%}' for k in KS) + f" | {E[E.g == g].amt.mean() / 1e8:.2f} |")
        D = M['最高組'] - M['最低組']
        L.append('| 高−低 | ' + ' | '.join(f'{D[:, k + k0].mean():+.2%}' for k in KS) + ' | |')
        L.append('| t | ' + ' | '.join(f'{nw_t(D[:, k + k0], max(1, abs(k) // 21)):.1f}' for k in KS) + ' | |')
        X = XS[(dates[XS[:, 0].astype(int)] >= pd.Timestamp(lo)) & (dates[XS[:, 0].astype(int)] <= pd.Timestamp(hi))]
        for c, nm in ((1, '60 日'), (2, '120 日')):
            v = X[:, c]; v = v[np.isfinite(v)]; top = np.sort(v)[-max(1, len(v) // 10):]
            sh = f'{top.sum() / v.sum():.0%}' if v.sum() > 0 else '（總超額 ≤ 0）'
            L.append(f"\n前 10 名個股 {nm}超額：P10 {np.percentile(v, 10):+.1%}、中位數 {np.median(v):+.1%}、P90 {np.percentile(v, 90):+.1%}、平均 {v.mean():+.1%}；前 10% 個股占總超額 {sh}")
        L.append('')
    L.append(f'（對照：強勢池股票日成交金額中位數平均 {amt_pool / 1e8:.2f} 億）\n')
    L += ['## B 部分：出場方式比較（前 10 名，每筆淨報酬已扣 0.77%）\n']
    score = {}
    for en, lo, hi in ERAS:
        E = B[(B.date >= lo) & (B.date <= hi)]
        L += [f'### {en}\n', '| 出場 | 筆數 | 平均淨報酬 | 勝率 | 平均持有日 | 平均持有期超額 | 年化每日超額 | 最大單筆虧損 | 前 10% 交易占總獲利 |', '|---|---|---|---|---|---|---|---|---|']
        for e in EXITS:
            x = E[E.e == e]; ann = x.ex.sum() / x.hold.sum() * 245
            score.setdefault(e, []).append(ann)
            top = np.sort(x.ret.values)[-max(1, len(x) // 10):]; sh = f'{top.sum() / x.ret.sum():.0%}' if x.ret.sum() > 0 else '（≤ 0）'
            L.append(f"| {e} | {len(x)} | {x.ret.mean():+.2%} | {(x.ret > 0).mean():.0%} | {x.hold.mean():.0f} | {x.ex.mean():+.2%} | {ann:+.1%} | {x.ret.min():+.1%} | {sh} |")
        L.append('')
    elig = {e: np.mean(v) for e, v in score.items() if all(z > 0 for z in v)}
    if elig:
        best = sorted(elig.items(), key=lambda kv: -kv[1])
        pick = best[0][0]
        if len(best) > 1 and best[0][1] - best[1][1] < 0.01 and SIMPLE[best[1][0]] < SIMPLE[best[0][0]]: pick = best[1][0]
        L += ['## 出場選定（事前固定規則）\n', '兩期年化每日超額皆 > 0 的出場（兩期平均）：' + '、'.join(f'{e} {v:+.1%}' for e, v in best),
              f'\n**選定：{pick}**\n']
    else:
        pick = None; L += ['## 出場選定\n', '**沒有任何出場在兩期年化每日超額皆 > 0。**\n']
    t120 = [A[(A.g == '前 10 名') & (A.date >= lo) & (A.date <= hi)].path.apply(lambda p: p[120 + k0]).mean() for _, lo, hi in ERAS]
    stop = not all(v > 0 for v in t120)
    L.append(f"**A 部分判讀：池外 R4 前 10 名 +120 日超額 {t120[0]:+.2%}｜{t120[1]:+.2%} → {'兩期皆不為正，建議停止' if all(v <= 0 for v in t120) else ('其中一期不為正，回報使用者' if stop else '兩期皆為正，可進 L-109 第一關')}**")
    head = f"池外前10名 +120 日 {t120[0]:+.2%}｜{t120[1]:+.2%}；出場選定 {pick}"
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
