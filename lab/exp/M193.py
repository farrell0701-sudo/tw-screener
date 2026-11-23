# M-193：M-192 確認集（偶數代碼，門檻沿用奇數代碼）。定義見 registry_M.md M-193。
import os
from pathlib import Path
os.environ['M192_PARITY'] = '0'
os.environ['M192_THR'] = (Path(__file__).resolve().parents[1] / 'results' / 'M192_thr.json').read_text()
from M192 import main as _m


def main():
    o = _m(); o['n_variants'] = 2; o['summary'] = o['summary'].replace('# M-192 結果', '# M-193 結果（M-192 確認集）'); return o
