# M-246：三支合成偶數代碼確認。定義見 registry_M.md M-246。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M245


def main():
    orig = M245.panel
    M245.panel = lambda parity: orig(0)
    out = M245.main(); M245.panel = orig
    s = out['summary'].replace('M-245 結果', 'M-246 結果（確認集）').replace('（奇數代碼）', '（偶數代碼）').replace('：不提名', '：確認不通過').replace('：提名', '：確認通過')
    return dict(summary=s, headline=out['headline'].replace('不提名', '確認不通過').replace('提名', '確認通過'), n_variants=0)
