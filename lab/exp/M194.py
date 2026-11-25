# M-194：X1、V4.1 訊號在 5／10／20 日內曾達 +1／2／3ATR 或 +5／10／15% 的比例（描述性）。定義見 registry_M.md M-194。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M190, M191

NS = (5, 10, 20); KS = (1, 2, 3); PS = (0.05, 0.10, 0.15)


def reach(te, jj, base, atr, HA, LA, T):
    out = {}
    for n in NS:
        ok = te + n < T; idx = np.nonzero(ok)[0]
        H = np.full((len(te), n), np.nan); L = np.full((len(te), n), np.nan)
        for d in range(n):
            H[idx, d] = HA[te[idx] + d, jj[idx]]; L[idx, d] = LA[te[idx] + d, jj[idx]]
        with np.errstate(all='ignore'):
            mx = np.nanmax(H, 1); mn = np.nanmin(L, 1)
        valid = ok & np.isfinite(mx) & np.isfinite(base) & np.isfinite(atr)
        for k in KS:
            hit = H >= (base + k * atr)[:, None]
            out[(n, f'+{k}ATR')] = np.where(valid, (mx >= base + k * atr).astype(float), np.nan)
            out[(n, f'+{k}ATR_d')] = np.where(valid & hit.any(1), hit.argmax(1) + 1, np.nan)
        for p in PS:
            hit = H >= (base * (1 + p))[:, None]
            out[(n, f'+{int(p * 100)}%')] = np.where(valid, (mx >= base * (1 + p)).astype(float), np.nan)
            out[(n, f'+{int(p * 100)}%_d')] = np.where(valid & hit.any(1), hit.argmax(1) + 1, np.nan)
        out[(n, '−1ATR')] = np.where(valid, (mn <= base - atr).astype(float), np.nan)
        out[(n, '−5%')] = np.where(valid, (mn <= base * 0.95).astype(float), np.nan)
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.load_sig('fix'); RKX = np.asarray(S['RKX']); RKV = np.asarray(S['RKV']); daily = S['daily'].reindex(dates)
    below = daily['below'].fillna(True).to_numpy(bool); off = daily['off'].fillna(True).to_numpy(bool)
    EX = M191.extra(P, F)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: i for i, k in enumerate(keys)}
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = dates.searchsorted(M190.HOLDOUT)
    sets = {}
    # (B) 全部訊號
    cx = np.isfinite(RKX); newx = cx & ~np.vstack([np.zeros((1, N), bool), cx[:-1]])
    for nm, M, g in (('X1 全部訊號', newx, below), ('V4.1 全部訊號', np.isfinite(RKV), off)):
        tt, jj = np.nonzero(M[t0:t1 - 1]); tt = tt + t0
        keep = odd[jj] & ~g[tt]; sets[nm] = (tt[keep] + 1, jj[keep])
    # (A) 實際交易
    B = sim.run_paths('fix', 100, sig='fix')
    for kind, nm in (('X', 'X1 實際交易'), ('V', 'V4.1 實際交易')):
        seen = set()
        for (r, tr, ex) in B[kind]:
            for x in tr:
                j = kidx.get(str(x[0])); d0 = pd.Timestamp(x[1])
                if j is None or not odd[j] or d0 >= M190.HOLDOUT or d0 < pd.Timestamp('2005-01-01'): continue
                seen.add((dates.get_loc(d0), j))
        a = np.array(sorted(seen)); sets[nm] = (a[:, 0], a[:, 1])
    L = ['# M-194 結果：X1、V4.1 在 N 日內曾達到的漲幅比例（奇數代碼、排除 2024-11-25 之後）\n',
         '進場＝訊號次一日開盤（實際交易用實際進場日）；以進場日起 N 個交易日內最高價判定；ATR＝訊號日 ATR20。括號內＝達到者的中位天數。\n']
    cols = [f'+{k}ATR' for k in KS] + [f'+{int(p * 100)}%' for p in PS]
    for era, lo, hi in M190.ERAS:
        L += [f'## {era}\n', '| 樣本 | 筆數 | N 日 | ' + ' | '.join(cols) + ' | 曾跌 −1ATR | 曾跌 −5% |', '|' + '---|' * (len(cols) + 5)]
        for nm, (te, jj) in sets.items():
            m = (dates[te] >= pd.Timestamp(lo)) & (dates[te] <= pd.Timestamp(hi)); te_, jj_ = te[m], jj[m]
            base = np.where(np.isfinite(OA[te_, jj_]), OA[te_, jj_], Cf[te_, jj_]); atr = EX['ATR20'][te_ - 1, jj_]
            R = reach(te_, jj_, base, atr, HA, LA, T)
            for n in NS:
                L.append(f"| {nm} | {len(te_):,} | {n} | " + ' | '.join(f"{np.nanmean(R[(n, c)]):.0%}（{np.nanmedian(R[(n, c + '_d')]):.0f}）" for c in cols)
                         + f" | {np.nanmean(R[(n, '−1ATR')]):.0%} | {np.nanmean(R[(n, '−5%')]):.0%} |")
        L.append('')
    # 參考：ATR 占價格比例
    for nm, (te, jj) in sets.items():
        a = EX['ATR20'][te - 1, jj] / Cf[te - 1, jj]
        L.append(f"- {nm}：訊號日 ATR20 ÷ 收盤 中位 {np.nanmedian(a):.1%}（1ATR 約等於這個百分比）")
    return dict(summary='\n'.join(L), headline='N 日內達標比例', n_variants=0)
