# M-248：大盤多空點翻空解剖。定義見 registry_M.md M-248。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from L155 import bull_line
from M195 import HOLDOUT

COST = 0.00385; FEAT = ['d', 'VX', 'SLP', 'MA200', 'CH60', 'WID']


def auc(y, x):
    m = np.isfinite(x); y, x = y[m], x[m]
    if y.sum() == 0 or (~y).sum() == 0: return np.nan
    r = pd.Series(x).rank().to_numpy(); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    P = load_panel(); dates = P['dates']
    s = P['index']['TAIEX_P'].reindex(dates).ffill().dropna(); d_ = s.index; px = s.to_numpy(); T = len(px)
    bear = bull_line(px, d_)
    m = d_.to_period('M'); first = pd.Series(np.arange(T), index=d_).groupby(m).min(); mi = {p: i for i, p in enumerate(first.index)}
    line = np.full(T, np.nan); wid = np.full(T, np.nan)
    for t in range(T):
        k = mi[m[t]]; a = first.iloc[k - 1] if k > 0 else first.iloc[0]; w = px[a:t + 1]; line[t] = (w.max() + w.min()) / 2; wid[t] = (w.max() - w.min()) / line[t]
    ch = pd.Series(np.abs(np.diff(px, prepend=np.nan)), index=d_)
    a14 = ch.rolling(14).mean().to_numpy(); a10 = ch.rolling(10).mean().to_numpy(); a60 = ch.rolling(60).mean().to_numpy()
    ma200 = pd.Series(px).rolling(200).mean().to_numpy()
    flips = np.r_[0, (np.diff(bear.astype(int)) != 0).astype(int)]; ch60 = pd.Series(flips).rolling(60).sum().shift(1).to_numpy()
    rows = []; t = 1
    while t < T:
        if bear[t] and not bear[t - 1]:
            e = t
            while e + 1 < T and bear[e + 1]: e += 1
            end = min(e + 1, T - 1)    # 翻多日
            R = px[end] / px[t] - 1; mdd = px[t:end + 1].min() / px[t] - 1
            rows.append(dict(t=t, date=d_[t], len=end - t, R=R, B=-0.8 * R - 0.8 * 2 * COST, mdd=mdd,
                             f20=px[min(t + 20, T - 1)] / px[t] - 1, f60=px[min(t + 60, T - 1)] / px[t] - 1,
                             d=(line[t] - px[t]) / a14[t], VX=a10[t] / a60[t], SLP=line[t] / line[t - 20] - 1 if t >= 20 else np.nan,
                             MA200=float(px[t] < ma200[t]) if np.isfinite(ma200[t]) else np.nan, CH60=ch60[t], WID=wid[t]))
            t = end + 1
        else: t += 1
    D = pd.DataFrame(rows); D['real'] = D.B > 0
    ERA = (('探索 2004–15', '2004-01-01', '2015-12-31'), ('確認 2016–24', '2016-01-01', str(HOLDOUT.date())))
    L = ['# M-248 結果：大盤多空點翻空解剖（真跌 vs 盤整誤判）\n']
    res = {}
    for era, lo, hi in ERA:
        E = D[(D.date >= lo) & (D.date < hi)]; ny = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
        Bs = E.B.sort_values(ascending=False); top = Bs.iloc[:max(1, int(len(Bs) * 0.1))].sum()
        L += [f'## {era}\n', f"翻空段數 {len(E)}（每年 {len(E) / ny:.1f}）；真跌 {E.real.sum()}（{E.real.mean():.0%}）、誤判 {(~E.real).sum()}；縮倉總效益 {E.B.sum():+.1%}（真跌 {E[E.real].B.sum():+.1%}、誤判 {E[~E.real].B.sum():+.1%}）；前 10% 段效益 {top:+.1%}\n",
              '| 組 | 段數 | 平均段長 | 段內指數報酬 | 段內最大跌幅 | 翻空後 20 日 | 翻空後 60 日 |', '|---|---|---|---|---|---|---|']
        for nm, g in (('真跌', E[E.real]), ('誤判', E[~E.real])):
            L.append(f"| {nm} | {len(g)} | {g.len.mean():.1f} | {g.R.mean():+.2%} | {g.mdd.mean():+.2%} | {g.f20.mean():+.2%} | {g.f60.mean():+.2%} |")
        L += ['', '| 特徵 | 真跌中位數 | 誤判中位數 | AUC（真跌） |', '|---|---|---|---|']
        r = {}
        for f in FEAT:
            a = auc(E.real.to_numpy(), E[f].to_numpy().astype(float)); r[f] = a
            L.append(f"| {f} | {E[E.real][f].median():.3f} | {E[~E.real][f].median():.3f} | {a:.2f} |")
        res[era] = (r, E); L.append('')
    (r1, E1), (r2, E2) = res[ERA[0][0]], res[ERA[1][0]]
    L += ['## 提名與過濾模擬（門檻＝探索期全部翻空段的特徵中位數；方向依探索期 AUC）\n', '| 特徵 | AUC 探索 | AUC 確認 | 提名 | 確認期濾掉誤判 | 確認期保留總效益 | 確認期保留段數 |', '|---|---|---|---|---|---|---|']
    noms = []
    for f in FEAT:
        a1, a2 = r1[f], r2[f]
        if not (np.isfinite(a1) and np.isfinite(a2)): continue
        up = a1 >= 0.5; s1, s2 = (a1 if up else 1 - a1), (a2 if up else 1 - a2)
        nom = s1 >= 0.60 and s2 >= 0.55
        thr = E1[f].median(); keep = (E2[f] > thr) if up else (E2[f] <= thr)
        if f == 'MA200': keep = (E2[f] == 1) if up else (E2[f] == 0)
        drop_w = (~keep & ~E2.real).sum() / max(1, (~E2.real).sum()); kept_B = E2[keep].B.sum() / E2.B.sum() if E2.B.sum() != 0 else np.nan
        L.append(f"| {f} | {a1:.2f} | {a2:.2f} | {'✓' if nom else ''} | {drop_w:.0%} | {kept_B:.0%} | {keep.sum()}/{len(E2)} |")
        if nom: noms.append(f)
    L.append('\n（保留總效益 > 100% 代表過濾後淨效益反而更高。）')
    head = f"翻空段：探索期誤判 {(~E1.real).mean():.0%}、確認期 {(~E2.real).mean():.0%}；提名特徵：{'、'.join(noms) or '無'}"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
