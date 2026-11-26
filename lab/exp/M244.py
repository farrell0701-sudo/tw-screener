# M-244：籌碼因子加入 X1 候選相關樹。定義見 registry_M.md M-244。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M239
from M127 import tdcc_factors
from L123 import daily_from_monthly
from chips import crowd
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import squareform

CH = {'B1': 'B1 千張大戶 1 月變化', 'B2': 'B2 千張大戶 3 月變化', 'B3': 'B3 400 張以上 1 月變化', 'B4': 'B4 散戶 1 月變化', 'B5': 'B5 股東人數 1 月變化率'}
LABX = {**M239.LAB, 'B1': '千張大戶1月變化', 'B2': '千張大戶3月變化', 'B3': '400張以上1月變化', 'B4': '散戶1月變化', 'B5': '股東人數1月變化率', 'TURN20': '周轉率20日', 'MUSE': '融資使用率'}
FAC = M239.FAC + list(CH) + ['TURN20', 'MUSE']
ERAS = (('2010–15', '2010-01-01', '2015-12-31'), ('2016–24', '2016-01-01', '2024-11-24'))


def extra(P, F, Tr, U):
    o = {k: np.asarray(v, np.float32) for k, v in M239.extra(P, F, Tr, U).items()}   # 先跑 lab/exp/M244_prep.py 產生籌碼快取
    for k in list(CH) + ['TURN20', 'MUSE']: o[k] = np.load(LAB / 'cache' / 'M244' / f'{k}.npy', mmap_mode='r')
    return o


def main():
    fac = [f for f in FAC if f != 'S']
    D = M200.panel(1, extra=extra, fac=fac)['X1 候選']
    out = LAB / 'results' / 'M244'; out.mkdir(parents=True, exist_ok=True)
    mats = {}; cov = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        cov[era] = E[FAC].notna().mean()
        acc = [g[FAC].rank().corr().to_numpy() for _, g in E.groupby('t') if len(g) >= 15]
        mats[era] = pd.DataFrame(np.nanmean(np.stack(acc), axis=0), index=FAC, columns=FAC)
    avg = (mats[ERAS[0][0]] + mats[ERAS[1][0]]) / 2; Z = {}
    for k, M in list(mats.items()) + [('兩期平均', avg)]:
        Dm = 1 - M.abs().fillna(0).to_numpy(); np.fill_diagonal(Dm, 0); Dm = (Dm + Dm.T) / 2
        Z[k] = linkage(squareform(Dm, checks=False), 'average'); M.round(3).to_csv(out / f'corr_{k}.csv.txt')
    stab = np.corrcoef(cophenet(Z[ERAS[0][0]]), cophenet(Z[ERAS[1][0]]))[0, 1]
    np.save(out / 'Z_avg.npy', Z['兩期平均']); cl = fcluster(Z['兩期平均'], t=0.5, criterion='distance')
    json.dump({'fac': FAC, 'lab': LABX, 'cl': cl.tolist(), 'stab': stab}, open(out / 'tree.json', 'w'), ensure_ascii=False)
    L = ['# M-244 結果：籌碼因子加入 X1 候選相關樹（奇數代碼，2010–24）\n', f'兩期樹形穩定度：{stab:.2f}\n',
         '籌碼因子覆蓋率（候選內非空比例）：' + '；'.join(f"{f} {cov[ERAS[0][0]][f]:.0%}／{cov[ERAS[1][0]][f]:.0%}" for f in list(CH) + ['TURN20', 'MUSE']) + '\n',
         '## |ρ| ≥ 0.5 切出的群\n']
    for c in sorted(set(cl)):
        L.append(f"- 群 {c}：" + '、'.join(f'{f}（{LABX[f]}）' for f, x in zip(FAC, cl) if x == c))
    L += ['', '## 籌碼因子與其他因子的最大相關（兩期平均）\n', '| 籌碼因子 | 最相關的因子 | ρ | 與營收群最大 |ρ| | 與價量最大 |ρ| |', '|---|---|---|---|---|']
    rev = ['R1', 'R4', 'R5', 'R6']; px = [f for f in M239.FAC if f not in rev]
    for f in list(CH) + ['TURN20', 'MUSE']:
        s = avg[f].drop(f); g = s.abs().idxmax()
        L.append(f"| {f} {LABX[f]} | {g} | {s[g]:+.2f} | {avg.loc[f, rev].abs().max():.2f} | {avg.loc[f, px].abs().max():.2f} |")
    return dict(summary='\n'.join(L), headline=f'樹形穩定度 {stab:.2f}，{len(set(cl))} 群', n_variants=0)
