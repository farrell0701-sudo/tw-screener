# M-102：X1「高浮盈後虧損」交易的共通點（A）；V4.1 訊號稀少時合併組合的後續報酬（B）。觀察式，探索集。定義見 registry_M.md M-102。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat, dvs_own
from chips import crowd
import signals, sim

START, END = pd.Timestamp('2005-01-01'), pd.Timestamp('2015-12-31')
EDGES = [-np.inf, 6, 11, 16, 22, np.inf]            # 同 M-101 的 V4.1 近 20 日訊號數切點（0–6、7–11、12–16、17–22、23+）


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; daily = S['daily']
    Tr, pool = dbg['Tr'], dbg['pool']
    prk = signals.prk
    dT = (pd.DataFrame(Tr) - pd.DataFrame(Tr).shift(10)).to_numpy()
    PT, PD = prk(Tr, pool), prk(dT, pool)
    CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64); VV = np.asarray(F['VALID'])
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    ATRP = np.asarray(S['ATRP'], np.float64); R20 = np.asarray(F['R20']); S60 = np.asarray(F['S60']); RRT = np.asarray(F['RRT'])
    DVS = dvs_own(CA.astype(np.float32), VV, 60); TURN, MUSE = crowd(P, F, 20)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    ix60 = ix.pct_change(60).to_numpy(); ix20 = ix.pct_change(20).to_numpy(); d200 = (ix / ix.rolling(200).mean() - 1).to_numpy()
    sid = P['meta']['stock_id'].astype(str); odd = (sid.str[-1].astype(int) % 2 == 1).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1
    kix = {k: j for j, k in enumerate(P['meta']['security_key'])}

    R = sim.run_paths('fix', 100, sig='fix')
    seen = {}
    for i in range(100):
        for (code, d0, d1, w, ret, hold, why, st, lk) in R['X'][i][1]:
            seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret, hold=hold, why=why))
    rows = []
    for v in seen.values():
        j = kix[v['code']]; t0 = int(dates.get_loc(v['d0'])); t1 = int(dates.get_loc(v['d1'])); s = t0 - 1
        if not odd[j] or t0 < t_lo or t1 > t_hi: continue
        ent = OA[t0, j]
        if not np.isfinite(ent) or ent <= 0: continue
        hp = Cf[t0:t1 + 1, j] / ent - 1; pk = int(np.nanargmax(hp))
        d = ATRP[s, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
        rows.append(dict(ret=v['ret'], why=v['why'], hold=v['hold'], mfe=float(hp[pk]), giveback=float(hp[pk] - v['ret']),
                         to_peak=pk, peak_to_exit=v['hold'] - pk, year=v['d0'].year,
                         T_ld百分位=PT[s, j], ΔT_ld百分位=PD[s, j], 漲幅20=R20[s, j], S60=S60[s, j], ATR百分比=ATRP[s, j], 停損寬度d=d,
                         DVS60=DVS[s, j], 回撤比60=RRT[s, j], 週轉率20=TURN[s, j], 融資使用率=MUSE[s, j],
                         指數60日=ix60[s], 指數距200MA=d200[s], 出場前指數20日=ix20[t1]))
    X = pd.DataFrame(rows)
    X['組'] = np.where(X.mfe >= 0.10, np.where(X.ret < 0, 'G1 高浮盈後虧損', 'G2 高浮盈後獲利'), 'G3 浮盈 < 10%')
    feats = ['T_ld百分位', 'ΔT_ld百分位', '漲幅20', 'S60', 'ATR百分比', '停損寬度d', 'DVS60', '回撤比60', '週轉率20', '融資使用率', '指數60日', '指數距200MA']
    paths = ['mfe', 'ret', 'giveback', 'to_peak', 'peak_to_exit', 'hold', '出場前指數20日']
    g = X.groupby('組')
    L = ['# M-102 結果（觀察式，探索集：奇數代碼、2005–15）\n', '## A：X1「高浮盈後虧損」交易的共通點\n',
         '分組：G1＝最高浮盈 ≥ 10% 且最終 < 0；G2＝最高浮盈 ≥ 10% 且最終 ≥ 0；G3＝最高浮盈 < 10%。表內為中位數。\n']
    cnt = g.size()
    L.append('| 項目 | ' + ' | '.join(f'{k}（{cnt[k]}）' for k in cnt.index) + ' |'); L.append('|---' * (len(cnt) + 1) + '|')
    L.append('| **進場前一日** |' + ' |' * len(cnt))
    for f in feats:
        L.append(f'| {f} | ' + ' | '.join(f'{g[f].median()[k]:.3f}' for k in cnt.index) + ' |')
    L.append('| **持有路徑** |' + ' |' * len(cnt))
    for f in paths:
        L.append(f'| {f} | ' + ' | '.join(f'{g[f].median()[k]:.3f}' for k in cnt.index) + ' |')
    # 出場原因與大盤急跌
    L.append('\n出場原因（筆數）：')
    for k in cnt.index:
        L.append(f'- {k}：' + '、'.join(f'{w} {n}' for w, n in X[X.組 == k].why.value_counts().items()))
    L.append('\n出場前 20 日指數跌超過 5% 的比例：' + '、'.join(f'{k} {(X[X.組 == k]["出場前指數20日"] < -0.05).mean():.0%}' for k in cnt.index))
    L.append('\nG1 進場年份：' + '、'.join(f'{y} {n}' for y, n in X[X.組.str.startswith('G1')].year.value_counts().sort_index().items()))
    L.append('\n全部交易的進場年份：' + '、'.join(f'{y} {n}' for y, n in X.year.value_counts().sort_index().items()))
    # G1 vs G2 的差：以 G2 的分布衡量（G1 中位數落在 G2 的第幾百分位）
    L.append('\nG1 中位數落在 G2 分布的百分位（遠離 50% 才有區分力）：')
    g1, g2 = X[X.組.str.startswith('G1')], X[X.組.str.startswith('G2')]
    L.append('、'.join(f'{f} {(g2[f] < g1[f].median()).mean():.0%}' for f in feats))

    # ---------- B ----------
    mixp = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(100)]
    pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    rC, rX, rV = pm(mixp), pm([R['X'][i][0] for i in range(100)]), pm([R['V'][i][0] for i in range(100)])
    eV = pm([R['V'][i][2] for i in range(100)]).reindex(dates)
    fwd = lambda r: ((1 + r).rolling(20).apply(np.prod, raw=True) - 1).shift(-20).reindex(dates)
    fC, fX, fV = fwd(rC), fwd(rX), fwd(rV)
    s20 = daily['n_signal'].rolling(20, min_periods=1).sum()
    ok = (np.arange(T) >= t_lo) & (np.arange(T) + 20 <= t_hi)
    b = pd.cut(s20[ok], EDGES, labels=['Q1 0–6', 'Q2 7–11', 'Q3 12–16', 'Q4 17–22', 'Q5 23+'])
    df = pd.DataFrame(dict(b=b.values, C=fC[ok].values, X=fX[ok].values, V=fV[ok].values, eV=eV[ok].values))
    t = df.groupby('b', observed=True).agg(n=('C', 'size'), C=('C', 'mean'), X=('X', 'mean'), V=('V', 'mean'), eV=('eV', 'mean'))
    L += ['\n## B：V4.1 近 20 日訊號數分組，後續 20 日報酬（新基準 100 路徑平均；2005–15；日資料重疊，只作描述）\n',
          '| 分組 | 日數 | 合併版 | X1 子帳戶 | V4.1 子帳戶 | V4.1 平均曝險 |', '|---|---|---|---|---|---|']
    for k, r in t.iterrows():
        L.append(f'| {k} | {int(r.n)} | {r.C:+.2%} | {r.X:+.2%} | {r.V:+.2%} | {r.eV:.0%} |')
    L.append(f"\n全部日子平均：合併 {df.C.mean():+.2%}、X1 {df.X.mean():+.2%}、V4.1 {df.V.mean():+.2%}。")
    return dict(summary='\n'.join(L), headline='觀察式描述完成', n_variants=0)
