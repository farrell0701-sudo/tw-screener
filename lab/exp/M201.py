# M-201：M-200 四個提名的偶數代碼確認集。定義見 registry_M.md M-201。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M200

TARGET = {('X1 候選', 'R5'), ('X1 候選', 'R6'), ('X1 候選', 'POS250'), ('V4.1 訊號', 'EXT')}


def main():
    L, vt = M200.analyze(M200.panel(0), 'M-201 結果：M-200 提名的偶數代碼確認集')
    hit = [v for v in vt if any(v.startswith(f'- {s}｜{f}：') for s, f in TARGET)]
    verdict = ['## 確認集判定（只判 4 個目標）\n'] + [f"- {s}｜{f}：{'通過' if any(h.startswith(f'- {s}｜{f}：') for h in hit) else '不通過'}" for s, f in sorted(TARGET)] + ['']
    return dict(summary='\n'.join(L[:1] + verdict + L[1:]), headline='通過：' + ('、'.join(f for s, f in sorted(TARGET) if any(h.startswith(f'- {s}｜{f}：') for h in hit)) or '無'), n_variants=0)
