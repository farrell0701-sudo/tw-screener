# M-165：營收股池內的因子發現（依 M-119；層＝池外營收池 A、全營收池 B；奇數代碼）。定義見 registry_M.md M-165。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M120
from M122 import fin_factors
from M127 import tdcc_factors

TWD = M120.TWD
NEW = ['S1 規模', 'S2 營收市值比', 'P1 取樣前60日超額', 'T1 T_ld', 'V1 ATR%', 'Q3 營益率年變化', 'Q5 毛利率創高']
CTRL = ['C1 R4 池內排序（對照）', 'C2 B4 散戶1月變化（對照）']
_CACHE = {}


def samp_prev(dates):
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    return {p - 1: (dates.get_loc(t), dates.get_loc(t) - 1) for p, t in samp.items() if dates.get_loc(t) > 0}


def r4_table():
    if 'R4' not in _CACHE: _CACHE['R4'] = M120.rev_factors()['R4 營收創高']
    return _CACHE['R4']


def shares_table(dates):
    """每個（取樣月−1）：可取得日 < 取樣日、季底距取樣日 ≤ 210 天的最新普通股股本 ÷ 10（股數）。"""
    fs = []
    for y in range(2002, 2027):
        f = TWD / 'derived' / 'financials' / f'{y}.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['security_key', 'statement', 'period_end', 'type', 'value', 'available_date'])
        d = d[(d.statement == 'balance') & (d.type == 'OrdinaryShare') & d.security_key.notna() & d.security_key.ne('')]
        fs.append(d)
    D = pd.concat(fs).sort_values('available_date'); D = D[D.value > 0]
    out = {}
    for m, (t, _) in samp_prev(dates).items():
        T0 = dates[t]; sub = D[D.available_date < T0]
        last = sub.groupby('security_key').tail(1); last = last[(T0 - last.period_end).dt.days <= 210]
        out[m] = last.set_index('security_key').value / 10
    return pd.DataFrame(out).T.sort_index()


def factors(dates):
    P = load_panel(); F = load_feat('AF'); T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, Tr = dbg['U'], dbg['Tr']; del S
    if Tr.shape[1] != N: full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
    C = np.asarray(P['C'], np.float64); CA = np.asarray(F['CA'], np.float64); ATR = np.asarray(F['ATR'], np.float64)
    VALID = np.asarray(F['VALID'])
    Cr = pd.DataFrame(np.where(C > 0, C, np.nan)).ffill(limit=5).to_numpy()
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    SH = shares_table(dates); shj = pd.Index(SH.columns.astype(str)).get_indexer(keys)
    rv = pd.read_parquet(TWD / 'derived' / 'month_revenue.parquet'); rv = rv[rv.security_key.notna() & rv.security_key.ne('')]
    rv['m'] = rv.period_end.dt.to_period('M')
    RW = rv.pivot_table(index='m', columns='security_key', values='revenue', aggfunc='last').astype(float)
    RW = RW.reindex(pd.period_range(RW.index.min(), RW.index.max(), freq='M')); RW[RW <= 0] = np.nan
    TTM = RW.rolling(12, min_periods=12).sum(); rj = pd.Index(TTM.columns.astype(str)).get_indexer(keys)
    R4 = r4_table(); r4j = pd.Index(R4.columns.astype(str)).get_indexer(keys)
    pick = lambda W, j, m: (np.where(j >= 0, W.loc[m].to_numpy(float)[np.clip(j, 0, None)], np.nan) if m in W.index else np.full(N, np.nan))
    rows = {k: {} for k in NEW[:5] + CTRL[:1]}
    for m, (t, tp) in samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2004-12-01'): continue
        sh = pick(SH, shj, m)
        with np.errstate(all='ignore'):
            mc = Cr[tp] * sh
            rows['S1 規模'][m] = np.log(np.where(mc > 0, mc, np.nan))
            rows['S2 營收市值比'][m] = pick(TTM, rj, m) / mc
            rows['P1 取樣前60日超額'][m] = CX[tp] - CX[tp - 60] if tp >= 60 else np.full(N, np.nan)
            rows['T1 T_ld'][m] = Tr[tp]
            rows['V1 ATR%'][m] = ATR[tp] / CA[tp]
        rows['C1 R4 池內排序（對照）'][m] = pick(R4, r4j, m)
    out = {}
    for k, d in rows.items():
        W = pd.DataFrame(d, index=keys).T.sort_index(); out[k] = W.loc[:, ~W.columns.duplicated()]
    fin = fin_factors(dates); out['Q3 營益率年變化'] = fin['Q3 營益率年變化']; out['Q5 毛利率創高'] = fin['Q5 毛利率創高']
    out['C2 B4 散戶1月變化（對照）'] = tdcc_factors(dates)['B4 散戶 1 月變化']
    return {k: out[k] for k in NEW + CTRL}


def layers(U, pool, dates, keys):
    R4 = r4_table(); j = pd.Index(R4.columns.astype(str)).get_indexer(keys)
    T, N = U.shape; A = np.zeros((T, N), bool); B = np.zeros((T, N), bool)
    for m, (t, _) in samp_prev(dates).items():
        if m not in R4.index: continue
        x = np.where(j >= 0, R4.loc[m].to_numpy(float)[np.clip(j, 0, None)], np.nan)
        for M, base in ((A, U[t] & ~pool[t]), (B, U[t])):
            s = base & np.isfinite(x)
            if s.sum() < 100: continue
            M[t] = s & (x >= np.nanpercentile(x[s], 90))
    return (('A 池外營收池', A, 5), ('B 全營收池', B, 5))


def verdict(G, fn):
    """M-165 提名判準（只看 A 層）。回傳（是否提名、各項結果、方向）。"""
    a, b = G.get((fn, 'A 池外營收池', '2005–15')), G.get((fn, 'A 池外營收池', '2016–26'))
    if a is None or b is None: return False, '資料不足', 0
    c1 = np.sign(a.ic60.iloc[0]) == np.sign(a.ic120.iloc[0]) == np.sign(b.ic60.iloc[0]) == np.sign(b.ic120.iloc[0])
    sa, sb = a.sp.iloc[0], b.sp.iloc[0]; ta, tb = abs(a.t.iloc[0]), abs(b.t.iloc[0])
    c2 = np.sign(sa) == np.sign(sb) and max(ta, tb) >= 2.0 and min(ta, tb) >= 1.0
    sgn = np.sign(sa + sb) or 1
    good, bad = (4, 0) if sgn > 0 else (0, 4)
    c3 = a.right.loc[good] >= 0.10 and b.right.loc[good] >= 0.10
    c4 = a.left.loc[good] < a.left.loc[bad] and b.left.loc[good] < b.left.loc[bad]
    flags = [c1, c2, c3, c4]
    return all(flags), '／'.join('✓' if f else '✗' for f in flags), int(sgn)


def main():
    R4 = r4_table()
    L, summ, G = M120.evaluate(1, None, fac_fn=factors, ref={'R4': R4}, layers=layers, detail=True)
    head = ['# M-165 結果：營收股池內的因子發現（奇數代碼；F 計數 50）\n',
            '層：A＝池外營收池（U 排除 T_ld 前 10%，R4 最高十分組）；B＝全營收池（U 內 R4 最高十分組）；層內五分組。\n',
            '## 提名判準（A 層；1 IC60/120 兩期同號／2 高−低兩期同號且 t ≥2、≥1／3 好組右尾 ≥10%／4 好組左尾 < 壞組）\n',
            '| 因子 | 方向 | 判準 1／2／3／4 | 提名 | A 高−低 60 日 05–15（t） | A 高−低 60 日 16–26（t） | 好組左尾／右尾 05–15 | 好組左尾／右尾 16–26 | 壞組左尾／右尾 05–15 | 壞組左尾／右尾 16–26 |',
            '|---|---|---|---|---|---|---|---|---|---|']
    nom = []
    for fn in NEW + CTRL:
        ok, fl, sg = verdict(G, fn)
        a, b = G.get((fn, 'A 池外營收池', '2005–15')), G.get((fn, 'A 池外營收池', '2016–26'))
        if a is None or b is None: head.append(f'| {fn} | — | 資料不足 | — | | | | | | |'); continue
        good, bad = (4, 0) if sg > 0 else (0, 4)
        ctrl = fn in CTRL
        if ok and not ctrl: nom.append(fn)
        head.append(f"| {fn} | {'高好' if sg > 0 else '低好'} | {fl} | {'（對照，不提名）' if ctrl else ('**提名**' if ok else '否')} | "
                    f"{a.sp.iloc[0]:+.2%}（{a.t.iloc[0]:.1f}） | {b.sp.iloc[0]:+.2%}（{b.t.iloc[0]:.1f}） | "
                    f"{a.left.loc[good]:.0%}／{a.right.loc[good]:.0%} | {b.left.loc[good]:.0%}／{b.right.loc[good]:.0%} | "
                    f"{a.left.loc[bad]:.0%}／{a.right.loc[bad]:.0%} | {b.left.loc[bad]:.0%}／{b.right.loc[bad]:.0%} |")
    head += ['', '## 摘要（IC 5／20／60／120 日；高−低 60 日超額）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    hl = f"營收池內因子：提名 {('、'.join(nom)) if nom else '無'}"
    return dict(summary='\n'.join(head + [''] + L), headline=hl, n_variants=0)
