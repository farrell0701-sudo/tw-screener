# M-143：週 EMA 長度高原（16–26 週），空頭辨識（以 M-108 下跌段為答案）。定義見 registry_M.md M-143。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from M108 import episodes

LENS = (16, 18, 20, 21, 22, 24, 26)


def stats(flag, down, epw, lo, hi):
    sel = (flag.index >= pd.Timestamp(lo)) & (flag.index <= pd.Timestamp(hi))
    f, dn = flag[sel], down[sel]
    rec = (f & dn).sum() / max(dn.sum(), 1); pre = (f & dn).sum() / max(f.sum(), 1)
    li, lo_ = [], []
    for pk, tr in epw:
        if not (pd.Timestamp(lo) <= pk <= pd.Timestamp(hi)): continue
        a = flag[flag.index > pk]; a = a[a]
        if len(a): li.append(int(((flag.index > pk) & (flag.index <= a.index[0])).sum()))
        b = flag[flag.index > tr]; b = b[~b]
        if len(b): lo_.append(int(((flag.index > tr) & (flag.index <= b.index[0])).sum()))
    sw = (f != f.shift()).sum() / (len(f) / 52)
    return dict(share=f.mean(), rec=rec, pre=pre, lin=np.median(li) if li else np.nan, lout=np.median(lo_) if lo_ else np.nan, sw=sw)


def main():
    P = load_panel(); dates = P['dates']
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); ma200 = ix.rolling(200).mean()
    wend = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max()
    wc = ix.loc[wend.values]; wm = ma200.loc[wend.values]; wc.index = wm.index = wend.values
    flags = {'G0 200日線下': (wc < wm)}
    for n in LENS:
        e = wc.ewm(span=n, adjust=False).mean(); flags[f'EMA{n} 下彎'] = e < e.shift(1)
    flags = {k: v.loc['2005':] for k, v in flags.items()}
    idx = flags['G0 200日線下'].index
    EP = episodes(ix.loc['2005':]); down = pd.Series(False, index=idx); epw = []
    for pk, tr, _ in EP: down[(idx > pk) & (idx <= tr)] = True; epw.append((pk, tr))
    L = ['# M-143 結果：週 EMA 長度高原（空頭辨識；答案＝M-108 下跌段，前視）\n']
    res = {}
    for era, lo, hi in (('全期', '2005-01-01', '2026-12-31'), ('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31')):
        L += [f'## {era}\n', '| 辨識器 | 標記週比例 | 召回率 | 精確率 | 首次標記（週，中位） | 低點後解除（週，中位） | 每年切換 |', '|---|---|---|---|---|---|---|']
        for k, f in flags.items():
            s = stats(f, down, epw, lo, hi); res[(era, k)] = s
            L.append(f"| {k} | {s['share']:.0%} | {s['rec']:.0%} | {s['pre']:.0%} | {s['lin']:.0f} | {s['lout']:.0f} | {s['sw']:.1f} |")
        L.append('')
    g = res[('全期', 'G0 200日線下')]
    L += ['## 高原判定（全期，事前標準）\n', '| 長度 | 解除 ≤ 7 | 召回 ≥ G0 | 精確 ≥ G0−3pt | 切換 ≤ 7 | 全部 |', '|---|---|---|---|---|---|']
    allok = True
    for n in LENS:
        s = res[('全期', f'EMA{n} 下彎')]
        c = [s['lout'] <= 7, s['rec'] >= g['rec'], s['pre'] >= g['pre'] - 0.03, s['sw'] <= 7]; allok &= all(c)
        L.append(f"| {n} 週 | " + ' | '.join('✓' if x else '✗' for x in c) + f" | {'✓' if all(c) else '✗'} |")
    L.append(f"\n**{'在高原內（16–26 全部達標）' if allok else '不在高原內'}**")
    return dict(summary='\n'.join(L), headline=f"週 EMA 高原：{'成立' if allok else '不成立'}", n_variants=0)
