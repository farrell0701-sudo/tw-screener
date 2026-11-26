# M-233：DH250 偶數代碼確認集。定義見 registry_M.md M-233。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M232


def main():
    L, vt = M200.analyze(M200.panel(0, extra=M232.extra, fac=M232.FAC), 'M-233 結果：DH250 偶數代碼確認集', fac=M232.FAC)
    ok = any(v.startswith('- X1 候選｜DH250：') for v in vt)
    return dict(summary='\n'.join(L[:1] + [f"## 確認集判定\n\n- X1 候選｜DH250：{'通過' if ok else '不通過'}\n"] + L[1:]), headline='DH250 ' + ('通過' if ok else '不通過'), n_variants=0)
