# M-151：A1 觸發擋、A1＋B1 同時轉好才恢復（狀態機；描述性篩選）。定義見 registry_M.md M-151。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M108 import episodes
from M143 import stats


def machine(a, b):
    out = np.zeros(len(a), bool); st = False
    for i in range(len(a)):
        if not st and a[i]: st = True
        elif st and not a[i] and not b[i]: st = False
        out[i] = st
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = np.nan; R[np.abs(R) > 0.5] = np.nan
    sh = lambda M: np.vstack([np.zeros((1, N), bool), M[:-1]])
    with np.errstate(all='ignore'):
        EW = (1 + pd.Series(np.nan_to_num(np.nanmean(np.where(sh(U), R, np.nan), 1)), index=dates)).cumprod()
        SP = (1 + pd.Series(np.nan_to_num(np.nanmean(np.where(sh(pool), R, np.nan), 1)), index=dates)).cumprod()
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    we = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max().values
    W = lambda s: pd.Series(s.loc[we].to_numpy(), index=we)
    wfall = lambda s, n: (lambda e: e < e.shift(1))(W(s).ewm(span=n, adjust=False).mean())
    A1 = wfall(EW, 21)
    D = {'G0 加權<200日線': W(ix < ix.rolling(200).mean()).astype(bool), 'A1 等權週EMA21下彎': A1}
    for n in (21, 16, 26):
        B1 = wfall(SP, n); D[f"A1→A1＋B1({n}){'（主）' if n == 21 else ''}"] = pd.Series(machine(A1.to_numpy(), B1.to_numpy()), index=we)
    D = {k: v.loc['2005':].astype(bool) for k, v in D.items()}
    idx = D['A1 等權週EMA21下彎'].index
    Rx = sim.run_paths('fix', 100, sig='fix'); rX = pd.concat([Rx['X'][i][0] for i in range(100)], axis=1).mean(axis=1).reindex(dates).fillna(0)
    eqX = (1 + rX).cumprod(); fX = (eqX.shift(-65) / eqX - 1).loc[idx]
    L = ['# M-151 結果：A1 觸發擋、A1＋B1 同時轉好才恢復（狀態機）\n']
    res = {}
    for ans, ser in (('(ii) 等權指數空頭', EW), ('(i) 加權指數空頭', ix)):
        EP = episodes(ser.loc['2005':]); down = pd.Series(False, index=idx); epw = []
        for pk, tr, _ in EP: down[(idx > pk) & (idx <= tr)] = True; epw.append((pk, tr))
        L.append(f'# 答案 {ans}\n')
        for era, lo, hi in (('全期', '2005-01-01', '2026-12-31'), ('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31')):
            full = era == '全期'
            L += [f'## {era}\n', '| 辨識器 | 標記週比例 | 召回率 | 精確率 | 首次標記 | 低點後解除 | 每年切換 |' + (' 2021 標記週 | 2023 下半標記週 | 標記時 X1 後13週 | 未標記時 |' if full else ''),
                  '|---|---|---|---|---|---|---|' + ('---|---|---|---|' if full else '')]
            for k, f in D.items():
                s = stats(f, down, epw, lo, hi); res[(ans, era, k)] = s
                ex = f" {int(f.loc['2021'].sum())} | {int(f.loc['2023-07':'2023-12'].sum())} | {fX[f].mean():+.1%} | {fX[~f].mean():+.1%} |" if full else ''
                L.append(f"| {k} | {s['share']:.0%} | {s['rec']:.0%} | {s['pre']:.0%} | {s['lin']:.0f} | {s['lout']:.0f} | {s['sw']:.1f} |" + ex)
            L.append('')
    g = res[('(ii) 等權指數空頭', '全期', 'G0 加權<200日線')]
    L += ['## 篩選（答案 (ii)、全期，事前標準）\n', '| 辨識器 | 解除 ≤ 7 | 切換 ≤ 4.5 | 召回 ≥ G0 | 精確 ≥ G0 | X1 標記期較差 | 全部 |', '|---|---|---|---|---|---|---|']
    allp = True
    for k in list(D)[2:]:
        s = res[('(ii) 等權指數空頭', '全期', k)]; f = D[k]
        c = [s['lout'] <= 7, s['sw'] <= 4.5, s['rec'] >= g['rec'], s['pre'] >= g['pre'], fX[f].mean() < fX[~f].mean()]; allp &= all(c)
        L.append(f"| {k} | " + ' | '.join('✓' if x else '✗' for x in c) + f" | {'✓' if all(c) else '✗'} |")
    L.append(f"\n**{'主版本與高原全部通過' if allp else '未全部通過'}**")
    return dict(summary='\n'.join(L), headline=f"A1→A1＋B1 狀態機篩選：{'通過' if allp else '未通過'}", n_variants=0)
