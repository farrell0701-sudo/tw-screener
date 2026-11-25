# M-183：M-182 X1（R6 頭部剔除月初弱勢 Q1）偶數代碼確認集，只測一次。定義見 registry_M.md M-183。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M182


def main():
    M182.build_orig = M182.build
    M182.build = lambda: M182.build_orig(parity=0)
    out = M182.main()
    s = out['summary'].replace('# M-182 結果：', '# M-183 結果（偶數代碼確認集）：').replace('（奇數代碼；F 計數 60）', '（偶數代碼；F 計數 60）')
    line = [l for l in s.splitlines() if l.startswith('- X1')][0]
    s = s.replace('## 判準\n', '## 判準（只判定 X1；K1、K2 僅供對照）\n')
    return dict(summary=s, headline='確認集 ' + line[2:], n_variants=0)
