# M-204：QUIET 合成在 X1 候選／V4.1 訊號內的增量（奇數代碼）。定義見 registry_M.md M-204。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M200


def add_q(out):
    for k, D in out.items():
        grp = 't' if k.startswith('X1') else D.date.dt.to_period('M')
        g = D.groupby(grp)
        D['QUIET'] = -(g.VC.rank(pct=True) + g.VX.rank(pct=True)) / 2
        # 對 POS250 也扣除：以 S 加 POS250 合成控制分數
        D['S_raw'] = D.S
    return out


def main(parity=1, title='M-204 結果：QUIET 在 X1 候選／V4.1 訊號內的增量（奇數代碼）'):
    out = add_q(M200.panel(parity))
    M200.FAC = ['QUIET', 'POS250', 'VC', 'VX']
    L, vt = M200.analyze(out, title)
    # 另報：S 與 POS250 合併為控制分數後 QUIET 的增量
    for k, D in out.items():
        grp = 't' if k.startswith('X1') else D.date.dt.to_period('M')
        D['S'] = D.groupby(grp).S_raw.rank(pct=True) + D.groupby(grp).POS250.rank(pct=True)
    M200.FAC = ['QUIET']
    L2, _ = M200.analyze(out, '附錄：控制分數＝S 百分位＋POS250 百分位時 QUIET 的增量')
    return dict(summary='\n'.join(L + [''] + L2), headline=f"提名 {len(vt)} 個：" + '；'.join(v.split('：')[0][2:] for v in vt), n_variants=0)
