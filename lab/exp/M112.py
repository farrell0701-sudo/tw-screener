# M-112：大盤自身 120/200 日線動能脈衝狀態與後續報酬（描述性）。定義見 registry_M.md M-112。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M107 import nw_t, NAMES

NP = 100


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    g = ix.rolling(120).mean() / ix.rolling(200).mean() - 1; dag = g.abs().diff(20)
    below = ix < ix.rolling(200).mean()
    R = sim.run_paths('fix', NP, sig='fix'); pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    rC = pm([sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NP)]).reindex(dates).fillna(0)
    rX = pm([R['X'][i][0] for i in range(NP)]).reindex(dates).fillna(0); rV = pm([R['V'][i][0] for i in range(NP)]).reindex(dates).fillna(0)
    eX = pm([R['X'][i][2] for i in range(NP)]).reindex(dates)
    fwd = lambda r, h: (np.log1p(r)[::-1].rolling(h).sum()[::-1].shift(-1)).pipe(np.expm1)   # t+1..t+h
    ixr = ix.pct_change().fillna(0)
    D = pd.DataFrame(dict(i20=fwd(ixr, 20), i60=fwd(ixr, 60), c20=fwd(rC, 20), x20=fwd(rX, 20), v20=fwd(rV, 20), ex=eX, below=below))
    L = ['# M-112 結果：大盤 120/200 日線動能脈衝狀態（描述性；新基準 100 路徑平均）\n',
         '後 N 日＝t+1 起；t 值為 NW（lag 20），大盤只有一條序列，t 值偏樂觀。\n']
    for c in (0.01, 0.005):
        st = pd.Series(-1, index=dates)
        f = g.notna() & dag.notna()
        for base, side in ((0, g >= 0), (3, g < 0)):
            m = f & side; st[m & (dag < -c)] = base; st[m & (dag.abs() <= c)] = base + 1; st[m & (dag > c)] = base + 2
        for lo, hi in (('2005-01-01', '2015-12-31'), ('2016-01-01', '2026-12-31')):
            sel = (dates >= lo) & (dates <= hi) & (st.values >= 0); S = st[sel]; d = D[sel]
            runs = (S != S.shift()).cumsum()
            L += [f'\n## c＝{c}，{lo[:4]}–{hi[2:4]}\n',
                  '| 狀態 | 天數占比 | 段落數 | 大盤<200MA 比例 | 後20日指數 | 後60日指數 | 後20日合併（t） | 後20日 X1 | 後20日 V4.1 | X1 曝險 |', '|---|---|---|---|---|---|---|---|---|---|']
            for k, nm in enumerate(NAMES):
                m = (S == k).values
                if not m.any(): L.append(f'| {nm} | 0% | 0 | | | | | | | |'); continue
                x = d[m]
                L.append(f"| {nm} | {m.mean():.0%} | {runs[S == k].nunique()} | {x.below.mean():.0%} | {x.i20.mean():+.2%} | {x.i60.mean():+.2%} | "
                         f"{x.c20.mean():+.2%}（{nw_t(x.c20.dropna(), 20):.1f}） | {x.x20.mean():+.2%} | {x.v20.mean():+.2%} | {x.ex.mean():.2f} |")
            L.append(f"| 全部 | 100% | | {d.below.mean():.0%} | {d.i20.mean():+.2%} | {d.i60.mean():+.2%} | {d.c20.mean():+.2%} | {d.x20.mean():+.2%} | {d.v20.mean():+.2%} | {d.ex.mean():.2f} |")
    return dict(summary='\n'.join(L), headline='大盤脈衝狀態描述完成', n_variants=0)
