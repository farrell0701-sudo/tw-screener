# M-235：VWG|DH 偶數代碼確認＋尾端分析。定義見 registry_M.md M-235。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M234
from M195 import ERAS


def main():
    out = M200.panel(0, extra=M234.extra, fac=M234.FAC)
    L, vt = M200.analyze(out, 'M-235 結果：VWG|DH 偶數代碼確認集', fac=M234.FAC)
    ok = any(v.startswith('- X1 候選｜VWG|DH：') for v in vt)
    D = out['X1 候選'].copy(); T = ['## 尾端分析：X1 候選依 VWG|DH 五分位（每期候選內；Q1 低＝較好）\n']
    heads = []
    for f in ('VWG|DH', 'VWG', 'DH250'):
        T.append(f'### {f}\n')
        for era, lo, hi in ERAS:
            E = D[(D.date >= lo) & (D.date <= hi)].dropna(subset=[f]).copy()
            E['q'] = E.groupby('t')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
            E = E.dropna(subset=['q'])
            g = E.groupby('q').agg(n=('e60', 'size'), e20=('e20', 'mean'), e60=('e60', 'mean'), win=('n20', lambda x: (x > 0).mean()),
                                   lft=('n20', lambda x: (x <= -0.10).mean()), rgt=('n20', lambda x: (x >= 0.20).mean()))
            mid = E[E.q.isin([2, 3, 4])].e60.mean()
            T += [f'**{era}**（{len(E):,} 筆）\n', '| 組 | 筆數 | 20 日超額 | 60 日超額 | 20 日淨勝率 | 左尾率（≤ −10%） | 右尾率（≥ +20%） |', '|---|---|---|---|---|---|---|']
            T += [f"| Q{int(q)} | {r['n']:,} | {r.e20:+.2%} | {r.e60:+.2%} | {r.win:.0%} | {r.lft:.1%} | {r.rgt:.1%} |" for q, r in g.iterrows()]
            dq5, dq1 = mid - g.loc[5, 'e60'], g.loc[1, 'e60'] - mid
            T.append(f'\n中間三組 60 日超額 {mid:+.2%}；Q5 落後 {dq5:+.2%}、Q1 領先 {dq1:+.2%}\n')
            if f == 'VWG|DH':
                kind = '左尾（剔除型）' if dq5 >= 1.5 * dq1 else ('頭部（排序型）' if dq1 >= 1.5 * dq5 else '兩者皆可')
                heads.append(f'{era} {kind}'); T.append(f'→ 判讀：{kind}\n')
    verdict = [f"## 確認集判定\n\n- X1 候選｜VWG|DH：{'通過' if ok else '不通過'}\n- 尾端：{'；'.join(heads)}\n"]
    return dict(summary='\n'.join(L[:1] + verdict + T + L[1:]), headline=f"VWG|DH {'通過' if ok else '不通過'}；{'；'.join(heads)}", n_variants=0)
