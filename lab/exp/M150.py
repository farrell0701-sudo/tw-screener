# M-150：空頭辨識的第二重確認——等權指數、強勢池健康度（描述性篩選）。定義見 registry_M.md M-150。
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


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = np.nan; R[np.abs(R) > 0.5] = np.nan
    sh = lambda M: np.vstack([np.zeros((1, N), bool), M[:-1]])
    with np.errstate(all='ignore'):
        ew = pd.Series(np.nan_to_num(np.nanmean(np.where(sh(U), R, np.nan), 1)), index=dates)
        sp = pd.Series(np.nan_to_num(np.nanmean(np.where(sh(pool), R, np.nan), 1)), index=dates)
    EW, SP = (1 + ew).cumprod(), (1 + sp).cumprod()
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    wend = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max(); we = wend.values
    W = lambda s: pd.Series(s.loc[we].to_numpy(), index=we)
    wfall = lambda s: (lambda e: e < e.shift(1))(W(s).ewm(span=21, adjust=False).mean())
    G0 = W(ix < ix.rolling(200).mean()).astype(bool)
    D = {'G0 加權<200日線': G0, 'E 加權週EMA21下彎': wfall(ix),
         'A1 等權週EMA21下彎': wfall(EW), 'A2 等權<200日線': W(EW < EW.rolling(200).mean()).astype(bool),
         'B1 強勢池週EMA21下彎': wfall(SP), 'B2 強勢池60日報酬<0': W(SP / SP.shift(60) - 1 < 0).astype(bool)}
    D['G0＆A2'] = D['G0 加權<200日線'] & D['A2 等權<200日線']; D['G0＆B2'] = D['G0 加權<200日線'] & D['B2 強勢池60日報酬<0']
    D = {k: v.loc['2005':].astype(bool) for k, v in D.items()}
    idx = D['G0 加權<200日線'].index
    Rx = sim.run_paths('fix', 100, sig='fix'); rX = pd.concat([Rx['X'][i][0] for i in range(100)], axis=1).mean(axis=1).reindex(dates).fillna(0)
    eqX = (1 + rX).cumprod(); fX = (eqX.shift(-65) / eqX - 1).loc[idx]
    L = ['# M-150 結果：空頭辨識的第二重確認——等權指數、強勢池健康度（描述性篩選）\n']
    res = {}
    for ans, ser in (('(i) 加權指數空頭', ix), ('(ii) 等權指數空頭', EW)):
        EP = episodes(ser.loc['2005':]); down = pd.Series(False, index=idx); epw = []
        for pk, tr, _ in EP: down[(idx > pk) & (idx <= tr)] = True; epw.append((pk, tr))
        L += [f'# 答案 {ans}（{len(EP)} 次：' + '、'.join(f'{a.date()}→{b.date()}' for a, b, _ in EP) + '）\n']
        for era, lo, hi in (('全期', '2005-01-01', '2026-12-31'), ('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31')):
            full = era == '全期'
            L += [f'## {era}\n', '| 辨識器 | 標記週比例 | 召回率 | 精確率 | 首次標記 | 低點後解除 | 每年切換 |' + (' 2021 標記週 | 2023 下半標記週 | 標記時 X1 後13週 | 未標記時 |' if full else ''),
                  '|---|---|---|---|---|---|---|' + ('---|---|---|---|' if full else '')]
            for k, f in D.items():
                s = stats(f, down, epw, lo, hi); res[(ans, era, k)] = s
                ex = f" {int(f.loc['2021'].sum())} | {int(f.loc['2023-07':'2023-12'].sum())} | {fX[f].mean():+.1%} | {fX[~f].mean():+.1%} |" if full else ''
                L.append(f"| {k} | {s['share']:.0%} | {s['rec']:.0%} | {s['pre']:.0%} | {s['lin']:.0f} | {s['lout']:.0f} | {s['sw']:.1f} |" + ex)
            L.append('')
    corr = pd.DataFrame({k: v.astype(float) for k, v in D.items()}).corr()
    L += ['## 與 G0 的相關（週標記）\n', '| 辨識器 | 與 G0 相關 |', '|---|---|'] + [f"| {k} | {corr.loc[k, 'G0 加權<200日線']:+.2f} |" for k in D]
    g = res[('(ii) 等權指數空頭', '全期', 'G0 加權<200日線')]
    L += ['\n## 篩選（答案 (ii)、全期，事前標準）\n', '| 辨識器 | 解除 ≤ 7 | 切換 ≤ 4.5 | 召回 ≥ G0 | 精確 ≥ G0 | X1 標記期較差 | 全部 |', '|---|---|---|---|---|---|---|']
    passed = []
    for k in list(D)[2:]:
        s = res[('(ii) 等權指數空頭', '全期', k)]; f = D[k]
        c = [s['lout'] <= 7, s['sw'] <= 4.5, s['rec'] >= g['rec'], s['pre'] >= g['pre'], fX[f].mean() < fX[~f].mean()]
        if all(c): passed.append(k)
        L.append(f"| {k} | " + ' | '.join('✓' if x else '✗' for x in c) + f" | {'✓' if all(c) else '✗'} |")
    L.append(f"\n**通過：{'、'.join(passed) if passed else '無'}**")
    return dict(summary='\n'.join(L), headline=f"第二重確認篩選：通過 {'、'.join(passed) if passed else '無'}", n_variants=0)
