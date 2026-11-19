# M-116：權值池（市值前 200 等權）轉弱廣度與反應（描述性）。定義見 registry_M.md M-116。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M113 import states
from M115 import fwd_ret, fwd_mdd

NP = 100; TOP = 200; ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    st, _ = states(F); ok = st >= 0; weak = ok & ((st == 0) | (st >= 3))
    SHR = pd.DataFrame(np.load(LAB / 'cache' / 'chips' / 'SHR.npy')).ffill().to_numpy()
    mc = np.asarray(P['C'], np.float64) * SHR; mc[~U] = np.nan
    rk = pd.DataFrame(mc).rank(axis=1, ascending=False, method='first').to_numpy(); C = rk <= TOP
    def share(den):
        n = den.sum(1).astype(float); n[n == 0] = np.nan; return pd.Series((weak & den).sum(1) / n, index=dates)
    WA, WB, WC = share(U & ok), share(pool & ok), share(C & ok)
    CA = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = CA / np.vstack([np.full((1, N), np.nan), CA[:-1]]) - 1
    R[~VALID] = np.nan; R[np.abs(R) > 0.5] = np.nan
    Csh = np.vstack([np.zeros((1, N), bool), C[:-1]]); Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        rCp = pd.Series(np.nan_to_num(np.nanmean(np.where(Csh, R, np.nan), 1)), index=dates)
        rAp = pd.Series(np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), 1)), index=dates)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = ix < ix.rolling(200).mean()
    Rs = sim.run_paths('fix', NP, sig='fix'); rS = pd.concat([sim.mix(Rs['X'][i][0], Rs['V'][i][0]) for i in range(NP)], axis=1).mean(axis=1).reindex(dates).fillna(0)
    Y = pd.DataFrame({'C池60': fwd_ret(rCp, 60), 'A池60': fwd_ret(rAp, 60), '指數60': fwd_ret(ix.pct_change().fillna(0), 60),
                      '指數60跌': fwd_mdd(ix, 60), '合併60': fwd_ret(rS, 60), '合併60跌': fwd_mdd((1 + rS).cumprod(), 60)})
    hdr = '| 組 | 天數 | C 池後60 | A 池後60 | 指數後60 | 指數大跌率 | 合併後60 | 合併60跌 |'
    row = lambda nm, x: f"| {nm} | {len(x)} | {x.C池60.mean():+.2%} | {x.A池60.mean():+.2%} | {x.指數60.mean():+.2%} | {(x.指數60跌 <= -0.10).mean():.0%} | {x.合併60.mean():+.2%} | {x.合併60跌.mean():.1%} |"
    L = ['# M-116 結果：權值池（市值前 200 等權）轉弱廣度與反應（描述性、不能提名）\n',
         f"權值池 C 平均股數 {np.nanmean(C.sum(1)[dates >= '2005']):.0f}；W＝上收斂＋任一「下」狀態比例。後 60 日＝t+1 起。\n"]
    for era, lo, hi in ERAS:
        sel = (dates >= lo) & (dates <= hi)
        L += [f'## {era}\n', '### 相關\n', '| | W_C | W_A（全市場） | W_B（強勢池） | 大盤<200MA |', '|---|---|---|---|---|']
        M = pd.DataFrame(dict(W_C=WC, W_A=WA, W_B=WB, 閘=below.astype(float)))[sel]
        L.append('| W_C | 1 | ' + ' | '.join(f"{M.W_C.corr(M[c]):+.2f}" for c in ('W_A', 'W_B', '閘')) + ' |')
        L.append(f"\n權值池等權 vs 全市場等權日報酬相關 {rCp[sel].corr(rAp[sel]):+.2f}；W_C 平均 {M.W_C.mean():.1%}、W_A {M.W_A.mean():.1%}。\n")
        for cond, cn in ((np.ones(sel.sum(), bool), '全部日子'), ((~below[sel]).values, '只看大盤 ≥ 200MA')):
            y = Y[sel][cond]; wc = WC[sel][cond]; wa = WA[sel][cond]; m = (wc.notna() & wa.notna()).values
            qc = pd.qcut(wc[m].rank(method='first'), 5, labels=False).values; qa = pd.qcut(wa[m].rank(method='first'), 5, labels=False).values
            ym = y[m]
            L += [f'### 依 W_C 分五組（{cn}）\n', hdr, '|---|---|---|---|---|---|---|---|']
            for k in range(5): L.append(row(f'Q{k + 1}', ym[qc == k]))
            L += [f'\n### 交叉（{cn}）\n', hdr, '|---|---|---|---|---|---|---|---|',
                  row('權值先轉弱（W_C 高、W_A 低）', ym[(qc >= 3) & (qa <= 1)]), row('中小先轉弱（W_C 低、W_A 高）', ym[(qc <= 1) & (qa >= 3)]),
                  row('兩者都高', ym[(qc >= 3) & (qa >= 3)]), row('兩者都低', ym[(qc <= 1) & (qa <= 1)]), '']
    return dict(summary='\n'.join(L), headline='權值池轉弱廣度描述完成', n_variants=0)
