# M-181：借券因子家族（M-119 評估；奇數代碼；2010–15、2016–26）。定義見 registry_M.md M-181。
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
import M120, M165
from M120 import TWD

ERAS = (('2010–15', '2010-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
FN = ['B1 借券強度', 'B2 借券費率', 'B3 借券加速']
_C = {}


def build(dates):
    if 'F' in _C: return _C['F']
    P = load_panel(); T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy(); sid = meta['stock_id'].astype(str).to_numpy()
    d = pd.concat(pd.read_parquet(f, columns=['date', 'stock_id', 'volume', 'fee_rate']) for f in sorted(glob.glob(str(TWD / 'staged' / 'TaiwanStockSecuritiesLending' / '*.parquet'))) if 'all' not in f)
    d['date'] = pd.to_datetime(d.date); d = d[d.volume > 0]
    d['fv'] = d.fee_rate * d.volume
    g = d.groupby(['date', 'stock_id'])[['volume', 'fv']].sum().reset_index()
    col = {s: i for i, s in enumerate(sid)}
    g = g[g.stock_id.isin(col)]; ti = dates.get_indexer(g.date); ok = ti >= 0; g = g[ok]; ti = ti[ok]; ci = g.stock_id.map(col).to_numpy()
    LV = np.zeros((T, N)); FV = np.zeros((T, N))
    np.add.at(LV, (ti, ci), g.volume.to_numpy(float)); np.add.at(FV, (ti, ci), g.fv.to_numpy(float))
    V = np.nan_to_num(np.asarray(P['V'], np.float64))
    cs = lambda A: np.vstack([np.zeros((1, N)), np.cumsum(A, axis=0)])
    cL, cF, cV = cs(LV), cs(FV), cs(V)
    win = lambda c, t, n: c[t + 1] - c[max(0, t + 1 - n)]
    out = {k: {} for k in FN}
    for m, (t, tp) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2009-06-01'): continue
        act = win(cL, tp, 250) > 0
        l20 = win(cL, tp, 20); v20 = win(cV, tp, 20); l60 = win(cL, tp, 60); f60 = win(cF, tp, 60)
        p60 = win(cL, tp - 20, 60) if tp >= 20 else np.zeros(N)
        with np.errstate(all='ignore'):
            out['B1 借券強度'][m] = np.where(act & (v20 > 0), l20 / v20, np.nan)
            out['B2 借券費率'][m] = np.where(act & (l60 > 0), f60 / l60, np.nan)
            out['B3 借券加速'][m] = np.where(act & (p60 > 0), l20 / (p60 / 3) - 1, np.nan)
    res = {}
    for k, dd in out.items():
        W = pd.DataFrame(dd, index=keys).T.sort_index(); res[k] = W.loc[:, ~W.columns.duplicated()]
    _C['F'] = res; return res


def main():
    L, summ = M120.evaluate(1, None, fac_fn=build, eras=ERAS)
    S1 = ['## 摘要（IC 5／20／60／120；高−低 60 日）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        S1.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    nom = []
    for fn in FN:
        for la in ('全市場', '強勢池'):
            r = {era: (ic, sp, t) for f_, l_, era, ic, sp, t in summ if f_ == fn and l_ == la}
            if len(r) < 2: continue
            a, b = r['2010–15'], r['2016–26']
            c1 = np.sign(a[0][2]) == np.sign(a[0][3]) == np.sign(b[0][2]) == np.sign(b[0][3])
            c2 = np.sign(a[1]) == np.sign(b[1]) and max(abs(a[2]), abs(b[2])) >= 2 and min(abs(a[2]), abs(b[2])) >= 1
            if c1 and c2: nom.append(f'{fn}（{la}）')
    head = ['# M-181 結果：借券因子家族（奇數代碼；2010–15、2016–26；F 計數 58）\n', f"依 M-119 判準（IC 同號＋高−低顯著；尾部見各表）提名候選：{'、'.join(nom) if nom else '無'}\n"]
    return dict(summary='\n'.join(head + S1 + [''] + L), headline='借券家族：' + ('、'.join(nom) if nom else '無提名'), n_variants=0)
