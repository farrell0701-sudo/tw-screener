# M-099：2015/6 漲跌幅放寬是否為因子翻轉的結構斷點（描述性＋判讀規則）。設計見 registry_M.md M-099 及「實作細節」補記。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat, dvs_own
import signals, sim

BPS = {'真斷點 2015-06': ('2012-06-01', '2015-06-01', '2018-05-31'),
       '安慰劑 2012-06': ('2009-06-01', '2012-06-01', '2015-05-29'),
       '安慰劑 2018-06': ('2015-06-01', '2018-06-01', '2021-05-31')}
REAL5 = ('2010-06-01', '2015-06-01', '2020-05-29')
ITEMS = ['1 V4.1 剔最陡', '2 X1 剔過熱', '3 X1 剔最陡', '4 V4.1 剔深回撤', '5 下檔波動占比', '6 MACD(20,40,9) 金叉', '7 T_ld 前 10% vs 可交易池']
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def macd_cross(CA, VALID):
    T, N = CA.shape; out = np.zeros((T, N), bool)
    for j in range(N):
        ii = np.flatnonzero(VALID[:, j])
        if len(ii) < 60: continue
        c = pd.Series(CA[ii, j].astype(float)); m = c.ewm(span=20, adjust=False).mean() - c.ewm(span=40, adjust=False).mean()
        sg = m.ewm(span=9, adjust=False).mean(); x = (m > sg) & (m.shift(1) <= sg.shift(1)); x.iloc[:60] = False
        out[ii, j] = x.values
    return out


def main():
    P = load_panel(); F = load_feat('AF'); d = P['dates']; M = P['meta']
    S = signals.compute('fix', P=P, F=F); D = S['_dbg']
    U, pool, hot, steep, sig0, sig = D['U'], D['pool'], D['hot'], D['steep'], D['sig0'], D['signal']
    deep = sig0 & ~sig
    CA = np.asarray(F['CA']); VALID = np.asarray(F['VALID'])
    dv = dvs_own(CA, VALID, 60); dvsf = np.nan_to_num(signals.prk(dv, pool)) > 0.8
    mx = macd_cross(CA, VALID)
    Cf = pd.DataFrame(CA.astype(float)).ffill().values
    fwd = np.full(Cf.shape, np.nan); fwd[:-20] = Cf[20:] / Cf[:-20] - 1; fwd[~VALID] = np.nan
    with np.errstate(all='ignore'):
        bench = np.nanmean(np.where(U, fwd, np.nan), axis=1)
    EXC = fwd - bench[:, None]
    groups = [(sig & steep, sig & ~steep), (pool & hot, pool & ~hot), (pool & steep, pool & ~steep), (deep, sig0 & ~deep),
              (pool & dvsf, pool & ~dvsf), (pool & mx, pool & ~mx), (pool, U)]
    last = M['stock_id'].str[-1]
    par = {'確認集（偶數）': last.isin(list('02468')).to_numpy(), '探索集（奇數）': last.isin(list('13579')).to_numpy()}

    def diff(A, B, lo, hi, cm):
        w = (d >= lo) & (d <= hi); a = EXC[w][:, cm][A[w][:, cm]]; b = EXC[w][:, cm][B[w][:, cm]]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        return (a.mean() - b.mean(), np.median(a) - np.median(b), len(a)) if len(a) and len(b) else (np.nan, np.nan, 0)

    flips = {}
    for pn, cm in par.items():
        log(f'## {pn}：「訊號組 − 基準組」後續 20 日超額報酬平均差（中位數差；訊號組筆數）\n')
        hdr = ['項目'] + [f'{k} 前' for k in BPS] + [f'{k} 後' for k in BPS]
        cols_ = []
        for bp, (a, m, b) in BPS.items(): cols_ += [(bp, a, (pd.Timestamp(m) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')), (bp, m, b)]
        log('| 項目 | ' + ' | '.join(f"{bp.split(' ')[0]}{bp.split(' ')[1]} {'前' if i % 2 == 0 else '後'}" for i, (bp, _, _) in enumerate(cols_)) + ' | 真斷點 5 年 前 | 後 |')
        log('|' + '---|' * (len(cols_) + 3))
        fl = {bp: 0 for bp in BPS}
        for it, (A, B) in zip(ITEMS, groups):
            vals = [diff(A, B, lo, hi, cm) for _, lo, hi in cols_]
            v5 = [diff(A, B, REAL5[0], '2015-05-29', cm), diff(A, B, REAL5[1], REAL5[2], cm)]
            for k, bp in enumerate(BPS):
                pre, post = vals[2 * k][0], vals[2 * k + 1][0]
                if np.isfinite(pre) and np.isfinite(post) and np.sign(pre) != np.sign(post): fl[bp] += 1
            f = lambda v: f'{v[0] * 100:+.2f}（{v[1] * 100:+.2f}；{v[2]}）'
            log(f'| {it} | ' + ' | '.join(f(v) for v in vals) + f' | {f(v5[0])} | {f(v5[1])} |')
        flips[pn] = fl
        log('\n翻轉項目數（平均差前後異號，同長度 3 年窗口）：' + '、'.join(f'{k} {v}/7' for k, v in fl.items()) + '\n')

    # 機制指標
    log('## 機制指標（全部股票）\n')
    LIST = np.asarray(P['TRD']) & ~np.asarray(P['EMG'])
    RC = np.asarray(P['C'], float); RO = np.asarray(P['O'], float); LU = np.asarray(P['LIMU'], float); LDp = np.asarray(P['LIMD'], float)
    with np.errstate(invalid='ignore'):
        upd = LIST & (RC >= LU * 0.999); dnd = LIST & (RC <= LDp * 1.001)
    Snp = signals.compute('fix', P=P, F=F, v_buyok_peek=False)['_dbg']['sig0']
    with np.errstate(invalid='ignore'):
        nxt_lu = np.zeros_like(Snp); nxt_lu[:-1] = RO[1:] >= LU[1:] * 0.999
    R20 = np.asarray(F['R20'])
    R = sim.run_paths('fix', 100)
    stops = []
    for i in range(100):
        for tr in R['X'][i][1]:
            if tr[6] != '移動停損': continue
            j = int(np.flatnonzero(M['security_key'].to_numpy() == tr[0])[0]); t0 = d.get_loc(tr[1]); t1 = d.get_loc(tr[2])
            hi = np.nanmax(Cf[t0:t1, j]) if t1 > t0 else np.nan; after = Cf[t1 + 1:t1 + 21, j]
            stops.append((tr[2], bool(len(after) and np.nanmax(after) > hi)))
    st = pd.DataFrame(stops, columns=['date', 'fake']).set_index('date').sort_index()

    def mech(lo, hi):
        w = (d >= lo) & (d <= hi)
        return dict(漲停占比=upd[w].sum() / LIST[w].sum(), 跌停占比=dnd[w].sum() / LIST[w].sum(),
                    次日開盤漲停=(Snp[w] & nxt_lu[w]).sum() / max(Snp[w].sum(), 1),
                    強勢池R20_P80=np.nanpercentile(np.where(pool[w], R20[w], np.nan), 80),
                    假停損率=st.loc[lo:hi, 'fake'].mean())
    rows = {}; delta = {}
    for bp, (a, m, b) in BPS.items():
        pre = mech(a, (pd.Timestamp(m) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')); post = mech(m, b)
        rows[f'{bp} 前'] = pre; rows[f'{bp} 後'] = post; delta[bp] = {k: post[k] - pre[k] for k in pre}
    tb = pd.DataFrame(rows).T
    log('| 窗口 | 漲停收盤占比 | 跌停收盤占比 | V4.1 事件次日開盤漲停 | 強勢池 20 日漲幅 P80 | X1 假停損率 |\n|---|---|---|---|---|---|')
    for k, r in tb.iterrows():
        log(f"| {k} | {r['漲停占比']:.2%} | {r['跌停占比']:.2%} | {r['次日開盤漲停']:.1%} | {r['強勢池R20_P80']:.1%} | {r['假停損率']:.1%} |")
    real = '真斷點 2015-06'; pl = [k for k in BPS if k != real]
    big = {k: abs(delta[real][k]) > max(abs(delta[p][k]) for p in pl) for k in delta[real]}
    log('\n真斷點前後變化是否大於兩個安慰劑：' + '、'.join(f"{k} {'是' if v else '否'}" for k, v in big.items()) + '\n')

    # 判讀（確認集）
    fc = flips['確認集（偶數）']; rf = fc[real]; pf = [fc[p] for p in pl]
    mech_ok = big['漲停占比'] or big['假停損率']
    if rf >= 4 and rf > max(pf) and mech_ok: verdict = '支持斷點假說'
    elif max(pf) >= rf: verdict = '否定（安慰劑翻轉數 ≥ 真斷點）'
    else: verdict = '不確定'
    log('## 判讀（確認集，依登記規則）\n')
    log(f"- 翻轉數：真斷點 {rf}/7；安慰劑 {pf[0]}/7、{pf[1]}/7（門檻：真斷點 ≥ 4 且多於兩個安慰劑）")
    log(f"- 機制：漲停占比變化{'大於' if big['漲停占比'] else '不大於'}安慰劑；假停損率變化{'大於' if big['假停損率'] else '不大於'}安慰劑")
    log(f"- 探索集翻轉數：" + '、'.join(f'{k} {v}/7' for k, v in flips['探索集（奇數）'].items()))
    log(f"\n**結論：{verdict}**")
    head = f'M-099 {verdict}：確認集真斷點翻轉 {rf}/7，安慰劑 {pf[0]}/7、{pf[1]}/7'
    return dict(summary='# M-099 2015/6 結構斷點\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=0, series={})
