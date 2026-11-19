# M-142：21 週 EMA 下彎 vs 200 日線，作為空頭辨識器（描述性、以 M-108 空頭為答案）。定義見 registry_M.md M-142。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M108 import episodes


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    ma200 = ix.rolling(200).mean()
    per = dates.to_period('W-FRI'); wend = pd.Series(dates, index=dates).groupby(per).max()
    wc = ix.loc[wend.values]; wm = ma200.loc[wend.values]; wc.index = wm.index = wend.values
    ema = wc.ewm(span=21, adjust=False).mean()
    W = pd.DataFrame(index=wc.index)
    W['G0 200日線下'] = wc < wm
    W['W1 EMA21 下彎'] = ema < ema.shift(1)
    W['W2 下彎且收盤在下'] = W['W1 EMA21 下彎'] & (wc < ema)
    W['W3 W2 連續 2 週'] = W['W2 下彎且收盤在下'] & W['W2 下彎且收盤在下'].shift(1, fill_value=False)
    W = W.loc['2005':]
    EP = episodes(ix.loc['2005':])
    down = pd.Series(False, index=W.index); epw = []
    for pk, tr, _ in EP:
        m = (W.index > pk) & (W.index <= tr); down[m] = True; epw.append((pk, tr))
    R = sim.run_paths('fix', 100, sig='fix'); rX = pd.concat([R['X'][i][0] for i in range(100)], axis=1).mean(axis=1).reindex(dates).fillna(0)
    eqX = (1 + rX).cumprod(); fX = (eqX.shift(-65) / eqX - 1).loc[W.index]
    fI = (ix.shift(-65) / ix - 1).loc[W.index]
    L = ['# M-142 結果：21 週 EMA 下彎 vs 200 日線（空頭辨識；答案＝M-108 下跌段，前視）\n',
         f"週數 {len(W)}；空頭 {len(EP)} 次；下跌段週占 {down.mean():.0%}。後 13 週＝約 65 個交易日。\n"]
    for era, lo, hi in (('全期', '2005', '2026'), ('2005–15', '2005', '2015'), ('2016–26', '2016', '2026')):
        sel = (W.index >= pd.Timestamp(lo)) & (W.index <= pd.Timestamp(hi + '-12-31'))
        L += [f'## {era}\n', '| 辨識器 | 標記週比例 | 召回率 | 精確率 | 首次標記（高點後週數，中位） | 解除（低點後週數，中位） | 每年切換 | 標記時指數後13週 | 未標記時 | 標記時 X1 後13週 | 未標記時 |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for c in W.columns:
            f = W[c][sel]; dn = down[sel]
            rec = (f & dn).sum() / max(dn.sum(), 1); pre = (f & dn).sum() / max(f.sum(), 1)
            lag_in, lag_out = [], []
            for pk, tr in epw:
                if not (pd.Timestamp(lo) <= pk <= pd.Timestamp(hi + '-12-31')): continue
                after = W[c][W.index > pk]; fl = after[after]
                if len(fl): lag_in.append(int((W.index[(W.index > pk)] <= fl.index[0]).sum()))
                aft = W[c][W.index > tr]; off = aft[~aft]
                if len(off): lag_out.append(int((W.index[(W.index > tr)] <= off.index[0]).sum()))
            sw = (f != f.shift()).sum() / (len(f) / 52)
            L.append(f"| {c} | {f.mean():.0%} | {rec:.0%} | {pre:.0%} | {np.median(lag_in) if lag_in else float('nan'):.0f} | {np.median(lag_out) if lag_out else float('nan'):.0f} | {sw:.1f} | "
                     f"{fI[sel][f].mean():+.1%} | {fI[sel][~f].mean():+.1%} | {fX[sel][f].mean():+.1%} | {fX[sel][~f].mean():+.1%} |")
        L.append('')
    L += ['## 各次空頭：首次標記（高點後週數）／解除（低點後週數）\n', '| 高點 → 低點 | ' + ' | '.join(W.columns) + ' |', '|---|' + '---|' * len(W.columns)]
    for pk, tr in epw:
        cells = []
        for c in W.columns:
            after = W[c][W.index > pk]; fl = after[after]; aft = W[c][W.index > tr]; off = aft[~aft]
            a = int((W.index[W.index > pk] <= fl.index[0]).sum()) if len(fl) else None
            b = int((W.index[W.index > tr] <= off.index[0]).sum()) if len(off) else None
            cells.append(f"{a}／{b}")
        L.append(f"| {pk.date()} → {tr.date()}（{int(((W.index > pk) & (W.index <= tr)).sum())} 週） | " + ' | '.join(cells) + ' |')
    return dict(summary='\n'.join(L), headline='空頭辨識器比較完成', n_variants=0)
