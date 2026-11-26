# M-239：X1 候選內因子相關樹狀圖。定義見 registry_M.md M-239。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M232, M238
from M195 import ERAS
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import squareform

FAC = ['S', 'T_ld', 'ΔT10', 'ΔT60', 'R1', 'R4', 'R5', 'R6', 'VC', 'VX', 'EXT', 'POS250', 'DH250', 'DSH', 'VWG', 'SL250', 'S60', 'R20', 'RG']
LAB = {'S': 'X1 排序分數', 'T_ld': 'T_ld 強度', 'ΔT10': 'T_ld 10日變化', 'ΔT60': 'T_ld 60日變化', 'R1': '營收單月年增', 'R4': '營收創高',
       'R5': '營收年增驚喜', 'R6': '營收近3月÷近12月', 'VC': '波動收縮 ATR20/60', 'VX': '量能 成交值20/60', 'EXT': '離 EMA20 延伸',
       'POS250': '250日位階', 'DH250': '距250日高點', 'DSH': '距高點天數', 'VWG': '離一年均價', 'SL250': '250日線斜率',
       'S60': '60日斜率', 'R20': '近20日漲幅', 'RG': '近3月−一年動能'}


def extra(P, F, Tr, U):
    a = M232.extra(P, F, Tr, U); b = M238.extra(P, F, Tr, U)
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill()
    pr = lambda x: pd.DataFrame(np.where(U, x, np.nan)).rank(axis=1, pct=True).to_numpy()
    rg = pr((C / C.shift(60) - 1).to_numpy()) - pr((C / C.shift(250) - 1).to_numpy())
    return {'DH250': a['DH250'], 'R20': a['R20'], 'DSH': a['DSH'], 'VWG': b['VWG'], 'SL250': b['SL250'], 'S60': b['S60'], 'RG': rg}


def main():
    fac = [f for f in FAC if f != 'S']
    D = M200.panel(1, extra=extra, fac=fac)['X1 候選']
    out = Path(__file__).resolve().parents[1] / 'results' / 'M239'; out.mkdir(parents=True, exist_ok=True)
    mats = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        acc = []; 
        for t, g in E.groupby('t'):
            if len(g) < 15: continue
            acc.append(g[FAC].rank().corr().to_numpy())
        mats[era] = pd.DataFrame(np.nanmean(np.stack(acc), axis=0), index=FAC, columns=FAC)
    avg = (mats[ERAS[0][0]] + mats[ERAS[1][0]]) / 2
    trees = {}
    for k, M in list(mats.items()) + [('兩期平均', avg)]:
        Dm = 1 - M.abs().to_numpy(); np.fill_diagonal(Dm, 0); Dm = (Dm + Dm.T) / 2
        Z = linkage(squareform(Dm, checks=False), 'average'); trees[k] = (Z, squareform(Dm, checks=False))
        M.round(3).to_csv(out / f'corr_{k}.csv.txt')   # 相關係數（非原始資料）
    c1 = cophenet(trees[ERAS[0][0]][0]); c2 = cophenet(trees[ERAS[1][0]][0]); stab = np.corrcoef(c1, c2)[0, 1]
    np.save(out / 'Z_avg.npy', trees['兩期平均'][0])
    cl = fcluster(trees['兩期平均'][0], t=0.5, criterion='distance')
    L = ['# M-239 結果：X1 候選內因子相關樹狀圖（奇數代碼）\n', f'兩期樹形穩定度（共表型相關）：{stab:.2f}\n', '## 在 |ρ| ≥ 0.5 處切出的群（兩期平均）\n']
    for c in sorted(set(cl)):
        mem = [f for f, x in zip(FAC, cl) if x == c]
        L.append(f"- 群 {c}：" + '、'.join(f'{f}（{LAB[f]}）' for f in mem))
    L += ['', '## 兩期平均相關（|ρ| ≥ 0.4 的配對）\n', '| 配對 | 2005–15 | 2016–24 |', '|---|---|---|']
    for i, a in enumerate(FAC):
        for b in FAC[i + 1:]:
            if abs(avg.loc[a, b]) >= 0.4: L.append(f'| {a}–{b} | {mats[ERAS[0][0]].loc[a, b]:+.2f} | {mats[ERAS[1][0]].loc[a, b]:+.2f} |')
    import json; json.dump({'fac': FAC, 'lab': LAB, 'cl': cl.tolist(), 'stab': stab}, open(out / 'tree.json', 'w'), ensure_ascii=False)
    return dict(summary='\n'.join(L), headline=f'樹形穩定度 {stab:.2f}，{len(set(cl))} 群', n_variants=0)
