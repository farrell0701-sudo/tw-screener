# M-196：停損區逾時（偶數代碼直接確認）。定義見 registry_M.md M-196。
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.environ['M195_PARITY'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M195

ERAS = M195.ERAS


def main():
    r = M195.main()
    D = pd.read_parquet('/tmp/M195_D_0.parquet').sort_values(['tid', 'd']).reset_index(drop=True)
    z = (D.stopd <= 0.25).astype(int)
    brk = (z == 0) | (D.tid != D.tid.shift())
    grp = brk.cumsum()
    run = z.groupby(grp).cumsum() * z
    allbig = {e: (D[D.era == e].rem >= 0.30).mean() for e in ERAS}
    rng = np.random.default_rng(0)
    out = ['# M-196 結果：停損區逾時（偶數代碼；到 2024-11-13；看過奇數代碼後的假設，直接確認）\n',
           '格式：觸發日換股差平均［95% 信賴區間］｜剩餘報酬｜大賺率｜死錢率｜崩跌率（觸發筆數）。全體持有日大賺率：' + '｜'.join(f'{allbig[e]:.0%}' for e in ERAS) + '\n',
           '| K | 2005–15 | 2016–24/11 |', '|---|---|---|']
    res = {}
    for K in (5, 3, 10):
        trig = D[(run == K)].drop_duplicates('tid')
        cells, ok = [], True
        for e in ERAS:
            x = trig[trig.era == e]
            if len(x) < 30: cells.append(f'—（{len(x)}）'); ok = False; continue
            v = x['diff'].to_numpy(); bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            big = (x.rem >= 0.30).mean()
            res[(K, e)] = (v.mean(), hi, big)
            cells.append(f"{v.mean():+.1%}［{lo:+.1%}, {hi:+.1%}］｜{x.rem.mean():+.1%}｜{big:.0%}｜{(x.rem.abs() <= 0.03).mean():.0%}｜{(x.rem <= -0.10).mean():.0%}（{len(x)}）")
        out.append(f"| {K}{'（主規格）' if K == 5 else ''} | {cells[0]} | {cells[1]} |")
    c = []
    for e in ERAS:
        m, hi, big = res.get((5, e), (np.nan, np.nan, np.nan))
        c.append((m <= -0.015, hi < 0, big <= allbig[e]))
    c4 = all(res.get((K, e), (np.nan,))[0] < 0 for K in (3, 10) for e in ERAS)
    ok = all(all(x) for x in c) and c4
    out += ['', '判準（K = 5）：' + '；'.join(f"{e}：(1) {'✓' if a else '✗'} (2) {'✓' if b else '✗'} (3) {'✓' if d else '✗'}" for e, (a, b, d) in zip(ERAS, c)) + f"；(4) {'✓' if c4 else '✗'}",
            f"\n**{'確認' if ok else '未確認'}**\n", '---', '## 附：M-195 偶數代碼描述（不算確認、不得提名）\n', r['summary']]
    return dict(summary='\n'.join(out), headline=('確認' if ok else '未確認') + '：停損區逾時', n_variants=1)
