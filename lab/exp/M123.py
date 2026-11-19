# M-123：Q5（毛利率創高）偶數代碼確認集（只測一次）。定義見 registry_M.md M-123。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M120, M122


def main():
    L, summ = M120.evaluate(0, ['Q5 毛利率創高'], fac_fn=M122.fin_factors, ref={'R4': M120.rev_factors()['R4 營收創高']})
    out = ['# M-123 結果：Q5（毛利率創高）確認集（偶數代碼，只測一次）\n', '| 期間 | 強勢池 高−低 60 日（t） | (a) | 強勢池 IC60 | (b) |', '|---|---|---|---|---|']
    ok = []
    for fn, la, era, ic, sp, t in summ:
        if la != '強勢池': continue
        a = sp > 0.02 and t >= 2; b = ic[2] > 0; ok.append(a and b)
        out.append(f"| {era} | {sp:+.2%}（{t:.1f}） | {'✓' if a else '✗'} | {ic[2]:+.3f} | {'✓' if b else '✗'} |")
    res = len(ok) == 2 and all(ok)
    out.append(f"\n**Q5：{'通過，可登記 L 系列' if res else '不通過，結案'}**\n")
    out += ['| 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        out.append(f"| {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(out + [''] + L), headline=f"Q5 {'通過' if res else '不通過'}", n_variants=0)
