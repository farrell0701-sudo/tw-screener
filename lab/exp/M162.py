# M-162：進場前漲幅與停損、營收資訊新舊（前月 vs 前前月）——X1 交易層描述。定義與判讀標準見 registry_M.md M-162。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M120 import rev_factors
from L123 import daily_from_monthly

NPATH = 100
VERS = {'新基準': 'fix', 'T1 營收先排（L-123）': 'L123_T1', 'U1 X1 高∧營收不差（L-124）': 'L124_U1'}
STOP = ('移動停損', '保本')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: j for j, k in enumerate(keys)}
    for s in VERS.values():
        assert (LAB / 'cache' / 'sig' / s / 'RKX.npy').exists(), s
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; del S
    r4d = daily_from_monthly(rev_factors(dates)['R4 營收創高'], dates, keys); r4p = signals.prk(r4d, pool)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    per = dates.to_period('M'); first = np.r_[True, per[1:] != per[:-1]]
    ms = np.maximum.accumulate(np.where(first, np.arange(T), 0))     # 當月第一個交易日索引
    D = {}
    for nm, sg in VERS.items():
        Rr = sim.run_paths('fix', NPATH, sig=sg); seen = {}
        for i in range(NPATH):
            for (code, d0, d1, wt, ret, hold, why, st, lk) in Rr['X'][i][1]:
                if d0 < pd.Timestamp('2005-01-01'): continue
                seen.setdefault((code, d0), (code, d0, d1, ret, hold, why))
        X = pd.DataFrame(list(seen.values()), columns=['code', 'd0', 'd1', 'ret', 'hold', 'why'])
        t0 = dates.get_indexer(pd.DatetimeIndex(X.d0)); t1 = dates.get_indexer(pd.DatetimeIndex(X.d1)); j = X.code.map(kidx).to_numpy()
        s = t0 - 1
        with np.errstate(all='ignore'):
            X['r5'] = Cf[s, j] / Cf[s - 5, j] - 1; X['r20'] = Cf[s, j] / Cf[s - 20, j] - 1
            me = ms[t0] - 1                                                  # 進場月份之前的最後交易日
            X['mtd'] = Cf[s, j] / Cf[me, j] - 1
        X['fresh'] = np.where(dates[t0].day > 10, '新（前月）', '舊（前前月）')
        X['fresh_sig'] = np.where(dates[s].day > 10, '新', '舊')
        X['fast'] = X.why.isin(STOP) & (X.hold <= 20)
        ok = t1 + 60 < T
        X['post60'] = np.where(ok, CX[np.minimum(t1 + 60, T - 1), j] - CX[t1, j], np.nan)
        X['r4p'] = r4p[s, j]
        D[nm] = X
    def row(x, lab):
        return (f"| {lab} | {len(x)} | {x.ret.mean():+.2%} | {(x.ret > 0).mean():.0%} | {x.fast.mean():.0%} | {x.why.isin(STOP).mean():.0%} | "
                f"{x.hold.mean():.0f} | {x.post60.mean():+.2%} |")
    HDR = ['| 組 | 筆數 | 平均報酬 | 勝率 | 20 日內停損／保本出場 | 停損＋保本 | 持有日 | 出場後 60 日超額 |', '|---|---|---|---|---|---|---|---|']
    L = ['# M-162 結果：進場前漲幅與停損、營收資訊新舊（X1 交易，100 路徑去重，2005 起，全部代碼）\n']
    fast_ok, fresh_ok = {}, {}
    for nm, X in D.items():
        L.append(f'# {nm}（{len(X)} 筆）\n')
        for col, lab in (('r20', '近 20 日漲幅'), ('r5', '近 5 日漲幅')):
            q = pd.qcut(X[col].rank(method='first'), 5, labels=False)
            L += [f'## 表一：依{lab}五分組（Q1 最低）\n'] + HDR
            for k in range(5):
                x = X[q == k]; L.append(row(x, f"Q{k + 1}（{x[col].median():+.1%}）"))
            if col == 'r20':
                lo, hi = X[q == 0], X[q == 4]
                fast_ok[nm] = (hi.fast.mean() - lo.fast.mean() >= 0.05) and (hi.ret.mean() < lo.ret.mean())
                L.append(f"\n最高組 − 最低組：快速出場率 {(hi.fast.mean() - lo.fast.mean()) * 100:+.1f}pt、平均報酬 {(hi.ret.mean() - lo.ret.mean()) * 100:+.2f}pt\n")
            else: L.append('')
        L += ['## 表二：營收資訊新舊（進場日 ≤ 10 日＝前前月）\n'] + HDR
        for g in ('新（前月）', '舊（前前月）'):
            L.append(row(X[X.fresh == g], g))
        n_, o_ = X[X.fresh == '新（前月）'], X[X.fresh == '舊（前前月）']
        fresh_ok[nm] = o_.ret.mean() < n_.ret.mean()
        L.append(row(X[X.fresh_sig == '新'], '並報：訊號日 > 10（新）')); L.append(row(X[X.fresh_sig == '舊'], '並報：訊號日 ≤ 10（舊）'))
        m = n_.mtd.median()
        L.append(row(n_[n_.mtd > m], f'新組內：本月以來漲幅高（> {m:+.1%}）')); L.append(row(n_[n_.mtd <= m], '新組內：本月以來漲幅低'))
        L += ['', '## 表三（只報告）：依進場時 R4 池內百分位\n'] + HDR
        L.append(row(X[X.r4p >= 0.8], 'R4 高（≥ 0.8）')); L.append(row(X[~(X.r4p >= 0.8)], '其他'))
        wc = X.why.value_counts(normalize=True)
        L.append('\n出場原因：' + '、'.join(f'{k} {v:.0%}' for k, v in wc.items()) + '\n')
    a = fast_ok['T1 營收先排（L-123）'] and fast_ok['U1 X1 高∧營收不差（L-124）']
    b = fresh_ok['T1 營收先排（L-123）'] and fresh_ok['U1 X1 高∧營收不差（L-124）']
    L += ['## 判讀（事前固定）\n', f"- 先漲多易停損（T1、U1 皆最高組快速出場率高 ≥ 5pt 且平均較低）：{'成立' if a else '不成立'}（" + '、'.join(f"{k[:2]} {'✓' if v else '✗'}" for k, v in fast_ok.items()) + '）',
          f"- 舊資訊較差（T1、U1 舊組平均皆低於新組）：{'成立' if b else '不成立'}（" + '、'.join(f"{k[:2]} {'✓' if v else '✗'}" for k, v in fresh_ok.items()) + '）']
    return dict(summary='\n'.join(L), headline=f"先漲多易停損{'成立' if a else '不成立'}；舊資訊較差{'成立' if b else '不成立'}", n_variants=0)
