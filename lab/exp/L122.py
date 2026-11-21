# L-122：V4.1 依風險調整部位大小（組合回測；L-118 例外 4）。判準見 registry_L.md L-122。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from v41ev import events

NPATH = 100; N_DSR = 568
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
CACHE = LAB / 'cache' / 'L122'
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, size=None):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig='fix', v_kw={'size': size} if size else None)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    expo = float(np.mean([R['V'][i][2].loc['2005':].mean() for i in range(NPATH)]))
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M, t=E.seg_table(M), tv=E.seg_table([x[0] for x in R['V']]), expo=expo)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: j for j, k in enumerate(keys)}
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; ATRP = np.asarray(S['ATRP'], np.float64); del S
    assert len(dbg['cols']) == N
    U, Tr, signal = dbg['U'], dbg['Tr'], dbg['signal']
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    # m：探索集（奇數 2005–15）V4.1 事件的 ATR% 中位數（訊號日）
    tt, jj = np.nonzero(signal); okm = (dates[tt] >= '2005-01-01') & (dates[tt] <= '2015-12-31') & odd[jj]
    m = float(np.nanmedian(ATRP[tt[okm], jj[okm]]))
    # B6 日資料：0.5 × cdf(A6) + 0.5 × (1 − cdf(A1))，cdf 取探索集事件分布（同 M-156）
    X = events(); ex = X[(X.par == 1) & (X.year <= 2015)]
    s6 = np.sort(ex.A6.to_numpy()); s1 = np.sort(ex.A1.to_numpy())
    A6 = signals.prk(Tr, U)
    A1 = np.full((T, N), np.nan)
    for j in range(N):
        ii = np.flatnonzero(VALID[:, j])
        if len(ii) < 25: continue
        m25 = pd.Series(CA[ii, j]).rolling(25).mean().to_numpy(); A1[ii, j] = CA[ii, j] / m25 - 1
    cdf = lambda s, x: np.searchsorted(s, np.nan_to_num(x, nan=np.nan), side='right') / len(s)
    with np.errstate(all='ignore'):
        B6 = 0.5 * np.where(np.isfinite(A6), cdf(s6, A6), np.nan) + 0.5 * (1 - np.where(np.isfinite(A1), cdf(s1, A1), np.nan))
    with np.errstate(all='ignore'): inv = m / ATRP
    SZ = {'S1': np.clip(inv, 0.5, 1.5), 'S2': np.clip(inv, 0.5, 1.0), 'S3': 0.5 + B6,
          'S1_0.67-1.33': np.clip(inv, 0.67, 1.33), 'S1_0.4-1.6': np.clip(inv, 0.4, 1.6)}
    CACHE.mkdir(parents=True, exist_ok=True); paths = {}
    for k, a in SZ.items():
        p = CACHE / f'{k}.npy'; np.save(p, a.astype(np.float32)); paths[k] = str(p)
    log(f'm（探索集 V4.1 事件 ATR% 中位數）＝{m:.4f}；探索集事件 {okm.sum()}\n')
    B = go('新基準'); O = {k: go(k, p) for k, p in paths.items()}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版（Calmar（年化）；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    log('\n## V4.1 單獨（Calmar／年化／回撤）與平均曝險\n')
    log('| 版本 | ' + ' | '.join(SEGS[:4]) + ' | 平均曝險 |\n|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tv'].loc[s, 'Calmar']:.2f}／{r['tv'].loc[s, '年化']:.1%}／{r['tv'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + f" | {r['expo']:.2f} |")
    log('\n## 交易層（V4.1，100 路徑合計，2005 起）\n')
    log('| 版本 | 平均 s | s P10／P50／P90 | 受現金限制比例 | 平均報酬 | 前 10% 交易占總獲利 |\n|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        rows = []
        for i in range(NPATH):
            for (code, d0, d1, wt, ret, hold, why, st, lk) in r['R']['V'][i][1]:
                if d0 < pd.Timestamp('2005-01-01'): continue
                rows.append((code, d0, wt, ret))
        D = pd.DataFrame(rows, columns=['code', 'd0', 'wt', 'ret'])
        if nm == '新基準': s = np.ones(len(D))
        else:
            t0 = dates.get_indexer(pd.DatetimeIndex(D.d0)) - 1; j = D.code.map(kidx).to_numpy()
            s = np.nan_to_num(SZ[nm][t0, j], nan=1.0)
        lim = np.mean(D.wt.to_numpy() < s / 4 - 1e-6)
        top = np.sort(D.ret.values)[-max(1, len(D) // 10):]; share = top.sum() / D.ret.sum() if D.ret.sum() > 0 else np.nan
        log(f"| {nm} | {s.mean():.2f} | {np.percentile(s, 10):.2f}／{np.percentile(s, 50):.2f}／{np.percentile(s, 90):.2f} | {lim:.0%} | {D.ret.mean():+.2%} | {share:.0%} |")
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); yr_ix = ix.groupby(dates.year).last().pct_change()
    m1 = E.path_mean(O['S1']['M'])
    yb = (1 + bpm).groupby(bpm.index.year).prod() - 1; y1 = (1 + m1).groupby(m1.index.year).prod() - 1
    log('\n## 每年合併報酬（S1 − 新基準；路徑平均）\n')
    log('| 年 | 新基準 | S1 | 差 | 加權指數 |\n|---|---|---|---|---|')
    for y in yb.index:
        if y < 2005: continue
        log(f"| {y}{'（空頭年）' if yr_ix.get(y, 0) < 0 else ''} | {yb[y]:+.1%} | {y1[y]:+.1%} | {y1[y] - yb[y]:+.1%} | {yr_ix.get(y, np.nan):+.1%} |")
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in SZ]
    ds = E.dsr(m1.loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    log(f"\nDSR（S1，全期日報酬，N＝{N_DSR}，試驗間變異數取本次五版本）：{ds:.2f}\n")
    mt = O['S1']['t']
    up = lambda k: all(O[k]['t'].loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    dn = lambda k: all(O[k]['t'].loc[s, 'Calmar'] < bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    c1 = up('S1'); c2 = Pv['S1'] >= 0.8; c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    pl = dn('S1_0.67-1.33') and dn('S1_0.4-1.6'); veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in SZ)
    ok = c1 and c2 and c3 and not pl and not veto
    log('## 判準（事前固定）\n')
    log(f"- S1 兩段 Calmar 都上升：{'成立' if c1 else '**不成立**'}（16-20 {mt.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；21-26 {mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P {Pv['S1']:.2f} ≥ 0.8：{'成立' if c2 else '**不成立**'}")
    log(f"- 05-15 回撤不加深 > 5pt：{'成立' if c3 else '**不成立**'}")
    log(f"- 高原：{'**失敗**' if pl else '成立'}；L-104 否決：{'**觸發**' if veto else '未觸發'}")
    log(f"\n**結論：{'通過（建議採用，待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-122 V4.1 依風險調整部位：{'通過' if ok else '不採用'}（S1 16-20 {mt.loc['2016–20', 'Calmar']:.2f}、21-26 {mt.loc['2021–26', 'Calmar']:.2f}、P {Pv['S1']:.2f}）"
    return dict(summary='# L-122 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(SZ),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
