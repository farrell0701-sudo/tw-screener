# M-133：「R4、E4 都要成立」偶數代碼確認集（只測一次）。定義見 registry_M.md M-133。
import sys
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M132
from M107 import nw_t


def main():
    groups = ['A', '0a', '0b', 'O', 'C']
    out = M132.run(0, groups)
    L = ['# M-133 結果：「R4、E4 都要成立」確認集（偶數代碼，只測一次）\n', '| 期間 | 後 60 日差（t） | > +1.5% 且 t ≥ 2 |', '|---|---|---|']
    ok = []
    for era, rec in out.items():
        d = rec['A']['d60']; m, t = np.mean(d), nw_t(d, 3); ok.append(m > 0.015 and t >= 2)
        L.append(f"| {era} | {m:+.2%}（{t:.1f}） | {'✓' if ok[-1] else '✗'} |")
    res = all(ok)
    L.append(f"\n**都要成立：{'通過' if res else '不通過，結案'}**\n")
    L += ['## 全部組別（偶數代碼，只報告）\n'] + M132.table(out, groups)
    return dict(summary='\n'.join(L), headline=f"都要成立 {'通過' if res else '不通過'}", n_variants=0)
