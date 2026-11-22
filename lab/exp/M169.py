# M-169：營收池訊號「上漲潛力耗盡」V2 偶數代碼確認集（只測一次）。定義見 registry_M.md M-169。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M168
from M107 import nw_t

V = 'V2 幅度（T_ld 第c日−進場日）'


def main():
    M168.PARITY = 0; M168.VARS = {V: -1}
    M168.main(); D = M168.LAST_D
    L = ['# M-169 結果：營收池訊號 V2「上漲潛力耗盡」｜偶數代碼確認集\n',
         '| 檢查點 | 期間 | 最弱組 60 日超額 | 其餘 | 差（t） | 最弱 左尾／右尾 | 其餘 左尾／右尾 | 最弱 剩餘到 120 日 | 其餘 | 最弱 最大跌幅 | 其餘 | 最弱 60 日內停損 | 其餘 | 訊號數 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    ver = {}
    for c in (20, 40, 60):
        res = {}
        for era in ('2005–15', '2016–26'):
            x = D[(D.c == c) & (D.era == era) & ~D.stopped].dropna(subset=[V, 'y']).copy()
            q10, q90 = x.y.quantile([.1, .9])
            x['w'] = x.groupby('m')[V].transform(lambda s: (-s).rank(pct=True, method='first') > 0.8 if len(s) >= 5 else pd.Series(False, index=s.index)).astype(bool)
            a, b = x[x.w], x[~x.w]
            g = x.groupby(['m', 'w']).y.mean().unstack(); dif = (g[True] - g[False]).dropna()
            tt = nw_t(dif.to_numpy(), 3)
            res[era] = dict(d=dif.mean(), t=tt, ra=(a.y >= q90).mean(), rb=(b.y >= q90).mean(), ma=a.rem.mean(), mb=b.rem.mean())
            L.append(f"| {c} | {era} | {a.y.mean():+.2%} | {b.y.mean():+.2%} | {dif.mean():+.2%}（{tt:.1f}） | {(a.y <= q10).mean():.0%}／{(a.y >= q90).mean():.0%} | {(b.y <= q10).mean():.0%}／{(b.y >= q90).mean():.0%} | "
                     f"{a.rem.mean():+.2%} | {b.rem.mean():+.2%} | {a.dd.mean():+.1%} | {b.dd.mean():+.1%} | {a.st60.mean():.0%} | {b.st60.mean():.0%} | {len(x)} |")
        r1, r2 = res['2005–15'], res['2016–26']
        c1 = r1['d'] < 0 and r2['d'] < 0 and min(r1['t'], r2['t']) <= -2.0 and max(r1['t'], r2['t']) <= -1.0
        c2 = r1['ra'] < r1['rb'] and r2['ra'] < r2['rb']
        c3 = r1['ma'] < r1['mb'] and r2['ma'] < r2['mb']
        ver[c] = (c1, c2, c3)
    ok = all(all(ver[c]) for c in (20, 40))
    mk = lambda f: '✓' if f else '✗'
    head = [L[0], f"**{'確認通過' if ok else '確認不通過'}**　c＝20：{'／'.join(mk(f) for f in ver[20])}；c＝40：{'／'.join(mk(f) for f in ver[40])}；c＝60（只報告）：{'／'.join(mk(f) for f in ver[60])}\n"]
    return dict(summary='\n'.join(head + L[1:]), headline=f"V2 上漲潛力耗盡確認集{'通過' if ok else '不通過'}", n_variants=0)
