# M-243：QUIET 殘差偶數代碼確認。定義見 registry_M.md M-243。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M242


def main():
    orig = M200.panel
    M200.panel = lambda parity, **kw: orig(0, **kw)   # 只換代碼奇偶，其餘同 M-242
    out = M242.main(); M200.panel = orig
    s = out['summary'].replace('M-242 結果', 'M-243 結果（確認集）').replace('（奇數代碼）', '（偶數代碼）').replace('提名', '通過', 1).replace('不提名', '不通過', 1)
    h = out['headline'].replace('不提名', '確認不通過').replace('提名', '確認通過')
    return dict(summary=s, headline=h, n_variants=0)
