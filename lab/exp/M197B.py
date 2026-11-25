# M-197B：M-197 高點錨定錯誤修正後重跑（奇數代碼）。定義見 registry_M.md M-197B。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M197


def main():
    M197.TAG = 'M-197B'; M197.CACHE = LAB / 'cache' / 'M197B'
    out = M197.main()
    out['summary'] = out['summary'].replace('# M-197 結果', '# M-197B 結果（高點錨定修正）', 1)
    return out
