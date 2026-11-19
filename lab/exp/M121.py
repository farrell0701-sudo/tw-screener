# M-121：月營收 R4、R5 偶數代碼確認集（只測一次）。定義見 registry_M.md M-121。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M120


def main():
    names = ['R4 營收創高', 'R5 年增驚喜']
    L, summ = M120.evaluate(0, names)
    out = ['# M-121 結果：月營收 R4、R5 確認集（偶數代碼，只測一次）\n', '## 判準\n',
           '| 因子 | 期間 | 強勢池 高−低 60 日（t） | (a) > +2% 且 t ≥ 2 | 強勢池 IC60 | (b) > 0 |', '|---|---|---|---|---|---|']
    res = {}
    for fn, la, era, ic, sp, t in summ:
        if la != '強勢池': continue
        a = sp > 0.02 and t >= 2; b = ic[2] > 0
        res.setdefault(fn, []).append(a and b)
        out.append(f"| {fn} | {era} | {sp:+.2%}（{t:.1f}） | {'✓' if a else '✗'} | {ic[2]:+.3f} | {'✓' if b else '✗'} |")
    heads = []
    for fn in names:
        ok = len(res.get(fn, [])) == 2 and all(res[fn]); heads.append(f"{fn[:2]} {'通過' if ok else '不通過'}")
        out.append(f"\n**{fn}：{'通過，可登記 L 系列' if ok else '不通過，結案'}**")
    out += ['\n## 摘要（IC 5／20／60／120；高−低 60 日）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日 |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        out.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%} |")
    return dict(summary='\n'.join(out + [''] + L), headline='；'.join(heads), n_variants=0)
