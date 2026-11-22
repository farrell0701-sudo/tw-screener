# M-166：Q5 毛利率創高在池外營收池的偶數代碼確認集（只測一次）。定義見 registry_M.md M-166。
import sys
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M120, M165

FN = 'Q5 毛利率創高'


def main():
    fac = lambda dates: {FN: M165.factors(dates)[FN]}
    L, summ, G = M120.evaluate(0, None, fac_fn=fac, ref={'R4': M165.r4_table()}, layers=M165.layers, detail=True)
    a, b = G[(FN, 'A 池外營收池', '2005–15')], G[(FN, 'A 池外營收池', '2016–26')]
    c1 = min(a.ic60.iloc[0], a.ic120.iloc[0], b.ic60.iloc[0], b.ic120.iloc[0]) > 0
    ta, tb = a.t.iloc[0], b.t.iloc[0]
    c2 = a.sp.iloc[0] > 0 and b.sp.iloc[0] > 0 and max(ta, tb) >= 2.0 and min(ta, tb) >= 1.0
    c3 = a.right.loc[4] >= 0.10 and b.right.loc[4] >= 0.10
    c4 = a.left.loc[4] < a.left.loc[0] and b.left.loc[4] < b.left.loc[0]
    ok = c1 and c2 and c3 and c4
    mark = lambda f: '✓' if f else '✗'
    H = ['# M-166 結果：Q5 毛利率創高｜池外營收池｜偶數代碼確認集\n',
         f"**{'確認通過' if ok else '確認不通過'}**（判準 1／2／3／4：{mark(c1)}／{mark(c2)}／{mark(c3)}／{mark(c4)}）\n",
         '| A 層 | IC60 | IC120 | 高−低 60 日（t） | 高組左尾／右尾 | 低組左尾／右尾 | 最低組 − 第2–5組平均（60 日超額） | 期數 |', '|---|---|---|---|---|---|---|---|']
    for era, g in (('2005–15', a), ('2016–26', b)):
        H.append(f"| {era} | {g.ic60.iloc[0]:+.3f} | {g.ic120.iloc[0]:+.3f} | {g.sp.iloc[0]:+.2%}（{g.t.iloc[0]:.1f}） | {g.left.loc[4]:.0%}／{g.right.loc[4]:.0%} | "
                 f"{g.left.loc[0]:.0%}／{g.right.loc[0]:.0%} | {g.e60.loc[0] - g.e60.loc[1:].mean():+.2%} | {g.n.iloc[0]} |")
    H.append('')
    for fn, la, era, ic, sp, t in summ:
        H.append(f"- {la}｜{era}：IC " + '／'.join(f'{v:+.3f}' for v in ic) + f"；高−低 60 日 {sp:+.2%}（t {t:.1f}）")
    return dict(summary='\n'.join(H + [''] + L), headline=f"Q5 營收池確認集{'通過' if ok else '不通過'}", n_variants=0)
