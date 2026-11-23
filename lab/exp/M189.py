# M-189：M-188 確認集（偶數代碼）。定義見 registry_M.md M-189。
import os
os.environ['M188_PARITY'] = '0'
from M188 import main as _m


def main():
    o = _m(); o['n_variants'] = 1; o['summary'] = o['summary'].replace('# M-188 結果', '# M-189 結果（M-188 確認集）'); return o
