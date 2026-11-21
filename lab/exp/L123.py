# L-123：X1 候選改為「營收創高（R4）先排、同階層比 X1 分數」（組合回測）。判準見 registry_L.md L-123。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M120 import rev_factors

NPATH = 100; N_DSR = 568
VARS = {'T1': ('tier', 10, 'R4'), 'T2': ('ew', 0, 'R4'), 'T3': ('tier', 10, 'RC'), 'T1_5層': ('tier', 5, 'R4'), 'T1_20層': ('tier', 20, 'R4')}
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def daily_from_monthly(W, dates, keys):
    """第 t 日：日期 > 10 用上個月，否則用上上個月（可取得日＝次月 10 日）。"""
    per = dates.to_period('M'); m = np.where(dates.day > 10, per - 1, per - 2)
    A = W.reindex(index=pd.PeriodIndex(m, freq='M')).reindex(columns=pd.Index(keys)).to_numpy(np.float32)
    return A


def go(name, sig):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig=sig)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]))


def trades(R):
    seen = {}
    for i in range(NPATH):
        for (code, d0, d1, wt, ret, hold, why, st, lk) in R['X'][i][1]:
            seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret, hold=hold, why=why))
    return pd.DataFrame(seen.values())


def conc(R, ind_of, dates, step=5):
    """每 step 日：X1 持股同產業最大檔數、HHI（100 路徑平均）。"""
    mx, hh = [], []
    days = dates[(dates >= '2005-01-01')][::step]
    for i in range(NPATH):
        tr = R['X'][i][1]
        if not tr: continue
        d0 = np.array([x[1] for x in tr], 'datetime64[ns]'); d1 = np.array([x[2] for x in tr], 'datetime64[ns]')
        ind = np.array([ind_of.get(x[0], '無') for x in tr], object)
        for d in days.values:
            h = ind[(d0 <= d) & (d1 > d)]
            if len(h) == 0: continue
            vc = pd.Series(h).value_counts().to_numpy(); mx.append(vc.max()); hh.append(((vc / vc.sum()) ** 2).sum())
    return np.mean(mx), np.mean(hh)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; keys = P['meta']['security_key'].astype(str).to_numpy()
    fac = rev_factors(dates); R1, R2, R4 = fac['R1 單月年增'], fac['R2 三月年增'], fac['R4 營收創高']
    sd = R1.rolling(12, min_periods=10).std(); E4 = R2 / sd.where(sd > 0)
    signals.EXTRA['R4'] = [daily_from_monthly(R4, dates, keys)]
    signals.EXTRA['RC'] = [signals.EXTRA['R4'][0], daily_from_monthly(E4, dates, keys)]
    # 基準訊號與 R4 池內百分位（必報用）
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); pool = S['_dbg']['pool']; cols = S['_dbg']['cols']; del S
    r4p = np.full((len(dates), len(keys)), np.nan, np.float32)
    r4p[:, cols] = signals.prk(signals.EXTRA['R4'][0][:, cols], pool)
    kidx = {k: j for j, k in enumerate(keys)}
    for nm, xr in VARS.items():
        signals.build(f'L123_{nm}', mode='fix', P=P, F=F, x_rev=xr)
    B = go('新基準', 'fix'); O = {nm: go(nm, f'L123_{nm}') for nm in VARS}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版（Calmar（年化）；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨（Calmar／年化／回撤）\n')
    log('| 版本 | ' + ' | '.join(SEGS[:4]) + ' |\n|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}／{r['tx'].loc[s, '年化']:.1%}／{r['tx'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + ' |')
    # 必報：交易層
    ind_of = dict(zip(keys, P['meta']['industry'].fillna('無').astype(str)))
    log('\n## X1 交易層（100 路徑去重；2005 起）\n')
    log('| 版本 | 筆數／年 | 平均持有日 | 停損＋保本出場比例 | 平均報酬 | 勝率 | 前 10% 交易占總獲利 | 持股 R4 池內百分位 | 同產業最大檔數 | 產業 HHI |\n|---|---|---|---|---|---|---|---|---|---|')
    TR = {}
    for nm, r in [('新基準', B), ('T1', O['T1']), ('T2', O['T2']), ('T3', O['T3'])]:
        X = trades(r['R']); X = X[X.d0 >= '2005-01-01']; TR[nm] = X
        yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
        t0 = dates.get_indexer(pd.DatetimeIndex(X.d0)) - 1; jj = X.code.map(kidx).to_numpy()
        X['r4p'] = r4p[t0, jj]
        pos = X.ret[X.ret > 0]; top = np.sort(X.ret.values)[-max(1, len(X) // 10):]
        share = top.sum() / X.ret.sum() if X.ret.sum() > 0 else np.nan
        mx, hh = conc(r['R'], ind_of, dates)
        npath_trades = np.mean([len([x for x in r['R']['X'][i][1] if x[1] >= pd.Timestamp('2005-01-01')]) for i in range(NPATH)]) / yrs
        log(f"| {nm} | {npath_trades:.1f} | {X.hold.mean():.0f} | {X.why.isin(['移動停損', '保本']).mean():.0%} | {X.ret.mean():+.2%} | {(X.ret > 0).mean():.0%} | {share:.0%} | {np.nanmean(X.r4p):.2f} | {mx:.2f} | {hh:.2f} |")
    log('\n## X1 交易依進場時 R4 池內百分位（高 ≥ 0.8）\n')
    log('| 版本 | 組 | 筆數 | 平均報酬 | 勝率 | 前 10% 交易占該組總獲利 |\n|---|---|---|---|---|---|')
    for nm, X in TR.items():
        for gn, mk in (('高', X.r4p >= 0.8), ('其他', ~(X.r4p >= 0.8))):
            x = X[mk]; top = np.sort(x.ret.values)[-max(1, len(x) // 10):]
            sh = f"{top.sum() / x.ret.sum():.0%}" if x.ret.sum() > 0 else '（總獲利 ≤ 0）'
            log(f"| {nm} | {gn} | {len(x)} | {x.ret.mean():+.2%} | {(x.ret > 0).mean():.0%} | {sh} |")
    # 每年合併報酬差
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    yr_ix = ix.groupby(dates.year).last().pct_change()
    m1 = E.path_mean(O['T1']['M'])
    yb = (1 + bpm).groupby(bpm.index.year).prod() - 1; y1 = (1 + m1).groupby(m1.index.year).prod() - 1
    log('\n## 每年合併報酬（T1 − 新基準；路徑平均）\n')
    log('| 年 | 新基準 | T1 | 差 | 加權指數 |\n|---|---|---|---|---|')
    for y in yb.index:
        if y < 2005: continue
        flag = '（空頭年）' if yr_ix.get(y, 0) < 0 else ''
        log(f"| {y}{flag} | {yb[y]:+.1%} | {y1[y]:+.1%} | {y1[y] - yb[y]:+.1%} | {yr_ix.get(y, np.nan):+.1%} |")
    # DSR
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    ds = E.dsr(m1.loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    log(f"\nDSR（T1，全期日報酬，N＝{N_DSR}，試驗間變異數取本次五版本）：{ds:.2f}\n")
    # 判準
    m = O['T1']['t']
    up = lambda k: all(O[k]['t'].loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    dn = lambda k: all(O[k]['t'].loc[s, 'Calmar'] < bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    c1 = up('T1'); c2 = Pv['T1'] >= 0.8; c3 = m.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    pl = dn('T1_5層') and dn('T1_20層'); veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and not pl and not veto
    log('## 判準（事前固定）\n')
    log(f"- T1 兩段 Calmar 都上升：{'成立' if c1 else '**不成立**'}（16-20 {m.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；21-26 {m.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P {Pv['T1']:.2f} ≥ 0.8：{'成立' if c2 else '**不成立**'}")
    log(f"- 05-15 回撤不加深 > 5pt：{'成立' if c3 else '**不成立**'}（{m.loc['2005–15', '回撤']:.1%} vs {bt.loc['2005–15', '回撤']:.1%}）")
    log(f"- 高原（5 層、20 層不得都兩段下降）：{'**失敗**' if pl else '成立'}；L-104 否決：{'**觸發**' if veto else '未觸發'}")
    others = [k for k in ('T2', 'T3') if up(k) and Pv[k] >= 0.8]
    log(f"- T2、T3 單獨達主判準：{'、'.join(others) if others else '無'}（只報告，不採用）")
    log(f"\n**結論：{'通過（建議採用，待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-123 X1 營收先排：{'通過' if ok else '不採用'}（T1 16-20 {m.loc['2016–20', 'Calmar']:.2f}、21-26 {m.loc['2021–26', 'Calmar']:.2f}、P {Pv['T1']:.2f}）"
    return dict(summary='# L-123 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
