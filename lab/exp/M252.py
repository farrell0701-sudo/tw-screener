# M-252：營收高＋安靜突破，2024-11-25 之後的保留期描述性檢查。定義見 registry_M.md M-252。
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
import M240, M251
from panel import load as load_panel
from feat import load_feat

HS = M240.HS; H0 = pd.Timestamp('2024-11-25')


def groups(E):
    G = {'A 全部突破': E, 'A｜營收高': E[E.rv == '高'], 'A｜營收低': E[E.rv == '低'], 'D 突破＋貼高＋量縮': E[E.hi & E.qt]}
    for k in ('高', '低'): G[f'D｜營收{k}'] = E[E.hi & E.qt & (E.rv == k)]
    return G


def table(E, title):
    L = [f'## {title}\n', '| 組 | 事件數 | ' + ' | '.join(f'{h}日' for h in HS) + ' | 20日淨勝率 | 左尾率 | 右尾率 | 高峰中位日 | 市值百分位中位 | 波動百分位中位 |',
         '|' + '---|' * (len(HS) + 8)]
    for nm, g in groups(E).items():
        if len(g) == 0: L.append(f'| {nm} | 0 |' + ' |' * (len(HS) + 6)); continue
        L.append(f'| {nm} | {len(g)} | ' + ' | '.join(f"{g[f'e{h}'].mean():+.2%}" for h in HS) +
                 f" | {(g.n20 > 0).mean():.0%} | {(g.n20 <= -0.10).mean():.1%} | {(g.n20 >= 0.20).mean():.1%} | {g.pk.median():.0f} | {g.mvp.median():.2f} | {g.vop.median():.2f} |")
    return L


def main():
    M240.HOLDOUT = pd.Timestamp('2100-01-01')
    P = load_panel(); F = load_feat('AF'); _, X = M251.factors(P, F); del F; gc.collect()
    ok = X['stock'][None, :] & np.isfinite(X['MV']) & (X['MV'] > 0)
    MVP = pd.DataFrame(np.where(ok, X['MV'], np.nan)).rank(axis=1, pct=True).to_numpy(np.float32)
    VOP = pd.DataFrame(np.where(ok, X['VOL'], np.nan)).rank(axis=1, pct=True).to_numpy(np.float32)
    del X, P; gc.collect()
    Ds = []
    for par in (1, 0):
        D = M240.build(par); D['par'] = par; Ds.append(D); gc.collect()
    D = pd.concat(Ds, ignore_index=True)
    D['mvp'] = MVP[D.t.to_numpy() - 1, D.j.to_numpy()]; D['vop'] = VOP[D.t.to_numpy() - 1, D.j.to_numpy()]
    Hd = D[D.date >= H0]; In = D[(D.date < H0) & (D.date >= '2016-01-01')]; In0 = D[D.date < '2016-01-01']
    L = ['# M-252 結果：營收高＋安靜突破（奇偶合併）\n',
         f'保留期事件日 {Hd.date.min():%Y-%m-%d}～{Hd.date.max():%Y-%m-%d}（需 120 日後資料）。超額＝相對強勢池等權。左尾率＝20 日淨報酬 ≤ −10%，右尾率 ≥ +20%。市值／波動百分位為全市場普通股內。\n']
    L += table(Hd, '保留期 2024-11-25 起') + [''] + table(In, '對照 2016–2024-11') + [''] + table(In0, '對照 2005–15') + ['']
    G = groups(Hd); a = lambda k, h: G[k][f'e{h}'].mean()
    c1 = a('A｜營收高', 60) - a('A｜營收低', 60); c2 = a('D｜營收高', 60) - a('A｜營收高', 60)
    c3 = (G['D 突破＋貼高＋量縮'].n20 <= -0.10).mean() - (G['A 全部突破'].n20 <= -0.10).mean()
    mo = Hd.groupby(Hd.date.dt.to_period('M'))
    L += ['## 事前預期對照（保留期）\n',
          f"- A｜營收高 − A｜營收低 60 日：{c1:+.2%}（預期 > 0）→ {'一致' if c1 > 0 else '翻轉'}",
          f"- D｜營收高 − A｜營收高 60 日：{c2:+.2%}（預期 ≥ 0）→ {'一致' if c2 >= 0 else '翻轉'}；D｜營收高 {len(G['D｜營收高'])} 筆",
          f"- D 左尾率 − A 左尾率：{c3:+.1%}（預期 < 0）→ {'一致' if c3 < 0 else '翻轉'}",
          f"- 事件月數 {mo.ngroups}；D｜營收高 事件分布於 {G['D｜營收高'].date.dt.to_period('M').nunique()} 個月"]
        # 逐筆列出 D｜營收高（樣本小，看是否集中）
    g = G['D｜營收高'].copy()
    if len(g):
        g['月'] = g.date.dt.to_period('M').astype(str)
        L += ['', '### D｜營收高 每月事件數與 60 日超額平均\n', '| 月 | 筆數 | 60日 | 120日 |', '|---|---|---|---|']
        for m, x in g.groupby('月'): L.append(f"| {m} | {len(x)} | {x.e60.mean():+.1%} | {x.e120.mean():+.1%} |")
    h = f"保留期：營收高−低 60 日 {c1:+.1%}、D｜營收高−A｜營收高 {c2:+.1%}、D 左尾率差 {c3:+.1%}（D｜營收高 {len(G['D｜營收高'])} 筆）"
    return dict(summary='\n'.join(L[:1] + [f'**{h}**\n'] + L[1:]), headline=h, n_variants=0)
