# M-129：B4（散戶 1 個月變化，反向）偶數代碼確認集（只測一次）。定義見 registry_M.md M-129。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'core'))
import M120
from M127 import tdcc_factors, ERAS


def main():
    L, summ = M120.evaluate(0, ['B4 散戶 1 月變化'], fac_fn=tdcc_factors, ref={'R4': M120.rev_factors()['R4 營收創高']}, eras=ERAS)
    out = ['# M-129 結果：B4 散戶 1 個月變化（反向）確認集（偶數代碼，只測一次）\n', '| 期間 | 強勢池 高−低 60 日（t） | (a) < −2% 且 t ≤ −2 | 強勢池 IC60 | (b) < 0 |', '|---|---|---|---|---|']
    ok = []
    for fn, la, era, ic, sp, t in summ:
        if la != '強勢池': continue
        a = sp < -0.02 and t <= -2; b = ic[2] < 0; ok.append(a and b)
        out.append(f"| {era} | {sp:+.2%}（{t:.1f}） | {'✓' if a else '✗'} | {ic[2]:+.3f} | {'✓' if b else '✗'} |")
    res = len(ok) == 2 and all(ok)
    out.append(f"\n**B4：{'通過，可登記 L 系列（須揭露提名放寬）' if res else '不通過，結案'}**\n")
    out += ['| 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        out.append(f"| {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(out + [''] + L), headline=f"B4 {'通過' if res else '不通過'}", n_variants=0)
