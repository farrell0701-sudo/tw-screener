# M-193：快速成功（5／10 日，+2ATR 與 +1ATR）對 V4.1 交易的分辨力（奇數代碼、排除最近兩年）。定義見 registry_M.md M-193。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M190, M191


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    EX = M191.extra(P, F)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: i for i, k in enumerate(keys)}
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    B = sim.run_paths('fix', 100, sig='fix')
    rows = {}
    for (r, tr, ex) in B['V']:
        for (code, d0, d1, w, ret, hold, why, st, lk) in tr:
            j = kidx.get(str(code)); d0 = pd.Timestamp(d0)
            if j is None or not odd[j] or d0 >= M190.HOLDOUT or d0 < pd.Timestamp('2005-01-01'): continue
            rows.setdefault((j, d0), (ret, max(int(hold), 1), why))
    D = pd.DataFrame([(j, d0, *v) for (j, d0), v in rows.items()], columns=['j', 'date', 'ret', 'hold', 'why'])
    te = dates.searchsorted(D.date.to_numpy()); jj = D.j.to_numpy()
    base = np.where(np.isfinite(OA[te, jj]), OA[te, jj], Cf[te, jj]); atr = EX['ATR20'][te - 1, jj]
    for nwin in (5, 10):
        for k in (1, 2):
            ok = np.full(len(D), np.nan); rN = np.full(len(D), np.nan)
            for i in range(len(D)):
                if te[i] + nwin >= T or not np.isfinite(atr[i]) or atr[i] <= 0: continue
                H = HA[te[i]:te[i] + nwin, jj[i]]; L = LA[te[i]:te[i] + nwin, jj[i]]
                up = H >= base[i] + k * atr[i]; dn = L <= base[i] - atr[i]
                iu = up.argmax() if up.any() else 99; idn = dn.argmax() if dn.any() else 99
                ok[i] = float(iu < idn and iu < 99)
                rN[i] = Cf[te[i] + nwin - 1, jj[i]] / base[i] - 1
            D[f'ok{nwin}_{k}'] = ok; D[f'r{nwin}'] = rN
    L = ['# M-193 結果：V4.1 交易的快速成功分辨力（奇數代碼、排除最近兩年；100 路徑去重）\n',
         '快速成功（N 日、+kATR）＝N 日內先碰進場 + k×ATR20（最高價）再碰進場 − 1ATR（最低價）；同日都碰＝失敗。\n']
    for era, lo, hi in M190.ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        L += [f"## {era}（{len(E):,} 筆；平均 {E.ret.mean():+.2%}、中位 {E.ret.median():+.2%}、勝率 {(E.ret > 0).mean():.0%}、持有 {E.hold.mean():.0f} 日）\n",
              '| 條件 | 成功比例 | 成功：平均／中位／勝率 | 未成功：平均／中位／勝率 | 勝率差 | 未成功者在第 N 日收盤的報酬（平均／中位） | 未成功者出場原因（前三） |', '|---|---|---|---|---|---|---|']
        for nwin in (5, 10):
            for k in (1, 2):
                c = f'ok{nwin}_{k}'; x = E.dropna(subset=[c]); s, f = x[x[c] == 1], x[x[c] == 0]
                wc = f.why.value_counts(normalize=True).head(3)
                L.append(f"| {nwin} 日、+{k}ATR | {x[c].mean():.0%} | {s.ret.mean():+.2%}／{s.ret.median():+.2%}／{(s.ret > 0).mean():.0%} | {f.ret.mean():+.2%}／{f.ret.median():+.2%}／{(f.ret > 0).mean():.0%} | "
                         f"{((s.ret > 0).mean() - (f.ret > 0).mean()) * 100:+.0f}pt | {f[f'r{nwin}'].mean():+.2%}／{f[f'r{nwin}'].median():+.2%} | " + '、'.join(f'{a} {b:.0%}' for a, b in wc.items()) + ' |')
        L.append('')
    return dict(summary='\n'.join(L), headline='V4.1 快速成功分辨力', n_variants=0)
