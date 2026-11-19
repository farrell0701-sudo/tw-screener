# M-147：補全週 EMA 閘門的三種設計（描述性篩選）。定義見 registry_M.md M-147。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from M108 import episodes
from M143 import stats


def main():
    P = load_panel(); dates = P['dates']
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); ma200 = ix.rolling(200).mean(); hi52 = ix.rolling(250, min_periods=200).max()
    wend = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max()
    wc, wm, wh = ix.loc[wend.values], ma200.loc[wend.values], hi52.loc[wend.values]
    for s in (wc, wm, wh): s.index = wend.values
    e = wc.ewm(span=21, adjust=False).mean(); E = e < e.shift(1); dd = wc / wh - 1
    run = E.astype(int).groupby((~E).cumsum()).cumsum()
    D = {'G0 200日線下': wc < wm, 'E 週EMA21下彎': E,
         'D1 連2週才擋（主）': run >= 2, 'D1 連3週才擋': run >= 3,
         'D2 跌幅≥8%（主）': E & (dd <= -0.08), 'D2 跌幅≥6%': E & (dd <= -0.06), 'D2 跌幅≥10%': E & (dd <= -0.10), 'D2 跌幅≥12%': E & (dd <= -0.12),
         'D5 兩者都看空': E & (wc < wm)}
    D = {k: v.loc['2005':].astype(bool) for k, v in D.items()}
    idx = D['G0 200日線下'].index
    EP = episodes(ix.loc['2005':]); down = pd.Series(False, index=idx); epw = []
    for pk, tr, _ in EP: down[(idx > pk) & (idx <= tr)] = True; epw.append((pk, tr))
    L = ['# M-147 結果：補全週 EMA 閘門的三種設計（描述性篩選；答案＝M-108 下跌段，前視）\n']
    res = {}
    for era, lo, hi in (('全期', '2005-01-01', '2026-12-31'), ('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31')):
        L += [f'## {era}\n', '| 設計 | 標記週比例 | 召回率 | 精確率 | 首次標記（週） | 低點後解除（週） | 每年切換 |' + (' 2021 標記週 | 2023 下半標記週 | 2025-05~06 再標記次數 |' if era == '全期' else ''),
              '|---|---|---|---|---|---|---|' + ('---|---|---|' if era == '全期' else '')]
        for k, f in D.items():
            s = stats(f, down, epw, lo, hi); res[(era, k)] = s
            extra = ''
            if era == '全期':
                n21 = int(f.loc['2021'].sum()); n23 = int(f.loc['2023-07':'2023-12'].sum()); x = f.loc['2025-05':'2025-06']; rf = int((x & ~x.shift(1, fill_value=bool(f.loc[:'2025-04'].iloc[-1]))).sum())
                extra = f' {n21} | {n23} | {rf} |'
            L.append(f"| {k} | {s['share']:.0%} | {s['rec']:.0%} | {s['pre']:.0%} | {s['lin']:.0f} | {s['lout']:.0f} | {s['sw']:.1f} |" + extra)
        L.append('')
    g = res[('全期', 'G0 200日線下')]
    def ok(k):
        s = res[('全期', k)]; return [s['lout'] <= 7, s['sw'] <= 4.5, s['rec'] >= g['rec'], s['pre'] >= g['pre']]
    L += ['## 篩選（全期，事前標準）\n', '| 設計 | 解除 ≤ 7 | 切換 ≤ 4.5 | 召回 ≥ G0 | 精確 ≥ G0 | 全部 |', '|---|---|---|---|---|---|']
    for k in list(D)[2:]:
        c = ok(k); L.append(f"| {k} | " + ' | '.join('✓' if x else '✗' for x in c) + f" | {'✓' if all(c) else '✗'} |")
    fam = {'D1': ['D1 連2週才擋（主）', 'D1 連3週才擋'], 'D2': ['D2 跌幅≥8%（主）', 'D2 跌幅≥6%', 'D2 跌幅≥10%', 'D2 跌幅≥12%'], 'D5': ['D5 兩者都看空']}
    passed = [f for f, ks in fam.items() if all(all(ok(k)) for k in ks)]
    L.append(f"\n**通過篩選（含高原）：{'、'.join(passed) if passed else '無'}**")
    return dict(summary='\n'.join(L), headline=f"閘門補全篩選：通過 {'、'.join(passed) if passed else '無'}", n_variants=0)
