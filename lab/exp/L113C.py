# L-113C：舊資料（old-data）中的興櫃價格是否為 2005–15 差距主因。試驗次數不累加。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
import old_panel, feat, signals, sim, evalu as E
from panel import load as load_panel

NPATH = 100
MODE = dict(bias1=False, bias2=True, bias3=True, amt='AMT20', ind='pk')
SEG = {'2005–15': ('2005-01-01', '2015-12-31')}
CAT = ['上市櫃', '興櫃', '不在 tw-data']
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def classify(P0):
    TW = load_panel(['EX', 'EMG']); twd = TW['dates']; tm = TW['meta']
    ti = twd.get_indexer(P0['dates']); okd = ti >= 0
    EX = np.asarray(TW['EX']); EMG = np.asarray(TW['EMG'])
    T, N = P0['C'].shape; cat = np.full((T, N), 2, np.int8)
    by = tm.groupby('stock_id').indices
    for j, c in enumerate(P0['meta']['stock_id']):
        ks = by.get(c)
        if ks is None: continue
        ex = np.zeros(T, bool); li = np.zeros(T, bool); em = np.zeros(T, bool)
        ex[okd] = EX[ti[okd]][:, ks].any(1)
        li[okd] = (EX[ti[okd]][:, ks] & ~EMG[ti[okd]][:, ks]).any(1)
        em[okd] = (EX[ti[okd]][:, ks] & EMG[ti[okd]][:, ks]).any(1)
        cat[em, j] = 1; cat[li, j] = 0
    return cat, int((~okd).sum())


def run(P, tag):
    t0 = time.time()
    F = feat.compute(P, 'AF_C'); S = signals.compute(MODE, P=P, F=F); c = sim.make_ctx(MODE, P, F, S, start='2005-01-01')
    X = [sim.runX(c, off=i % 20, seed=1000 + i) for i in range(NPATH)]
    V = [sim.runV(c, off=i % 20, seed=1000 + i) for i in range(NPATH)]
    M = [sim.mix(x[0], v[0]) for x, v in zip(X, V)]
    log(f'  （{tag}：{time.time() - t0:.0f} 秒）')
    return dict(F=F, S=S, X=X, V=V, M=M, t=E.seg_table(M, SEG).iloc[0], tx=E.seg_table([x[0] for x in X], SEG).iloc[0],
                tv=E.seg_table([v[0] for v in V], SEG).iloc[0])


def main():
    P0 = old_panel.build('old', 'list'); cat, nd = classify(P0)
    d = P0['dates']; w = (d >= '2005-01-01') & (d <= '2015-12-31')
    log(f'舊資料 {P0["C"].shape[1]} 檔；舊資料日期不在 tw-data 交易日曆：{nd} 天\n')
    B = run(P0, '基準：舊資料原樣')
    VAL = np.asarray(B['F']['VALID']); D = B['S']['_dbg']
    log('## 1. 三類在各層的比例（2005–15）\n')
    log('| 層 | ' + ' | '.join(CAT) + ' |\n|---|---|---|---|')
    for nm, m in (('有效格', VAL), ('可交易池', D['U']), ('X1 強勢池', D['pool']), ('V4.1 訊號', D['signal'])):
        mm = m[w]; cc = cat[w][mm]; n = len(cc)
        log(f'| {nm}（{n:,}） | ' + ' | '.join(f'{(cc == k).mean():.1%}' for k in range(3)) + ' |')
    log('\n## 2. 交易中進場日屬各類的比例與報酬貢獻（2005–15 進場，100 路徑合計）\n')
    log('| 策略 | 類別 | 筆數比例 | 平均報酬 | 貢獻（初始資金比例，路徑平均） |\n|---|---|---|---|---|')
    codes = {c: j for j, c in enumerate(P0['meta']['security_key'])}
    for k, wt, nm in (('X', 0.6, 'X1'), ('V', 0.4, 'V4.1')):
        rows = [(cat[d.get_loc(x[1]), codes[x[0]]], x[3] * x[4] * wt / NPATH, x[4]) for p in B[k] for x in p[1]
                if pd.Timestamp('2005-01-01') <= x[1] <= pd.Timestamp('2015-12-31')]
        a = pd.DataFrame(rows, columns=['cat', 'contrib', 'ret'])
        for ci in range(3):
            s_ = a[a.cat == ci]
            if len(s_): log(f'| {nm} | {CAT[ci]} | {len(s_) / len(a):.1%} | {s_.ret.mean():+.1%} | {s_.contrib.sum():+.2f}（占 {s_.contrib.sum() / a.contrib.sum():.0%}） |')
    P1 = old_panel.build('old', 'list'); P1['TRD'] = P1['TRD'] & (cat != 1); R1 = run(P1, 'R1：剔除興櫃格')
    P2 = old_panel.build('old', 'list'); P2['TRD'] = P2['TRD'] & (cat == 0); R2 = run(P2, 'R2：只留上市櫃格')
    log('\n## 3. 重跑（2005–15，合併版 100 路徑中位數）\n')
    log('| 版本 | 年化 | 回撤 | Calmar | X1 Calmar | V4.1 Calmar | V4.1 訊號數 |\n|---|---|---|---|---|---|---|')
    for nm, r in (('舊資料原樣（L-113B）', B), ('R1 剔除興櫃格', R1), ('R2 只留上市櫃格', R2)):
        t = r['t']; ev = int(r['S']['daily'].loc['2005':'2015', 'n_signal'].sum())
        log(f"| {nm} | {t['年化']:.1%} | {t['回撤']:.1%} | {t['Calmar']:.2f} | {r['tx']['Calmar']:.2f} | {r['tv']['Calmar']:.2f} | {ev} |")
    log('| 新基準（L-113 修正模式，tw-data） | 22.9% | −25.2% | 0.89 | 0.66 | 0.67 | 1,958 |')
    drop = B['t']['Calmar'] - R1['t']['Calmar']
    main_cause = drop >= 0.165
    log(f"\n## 判讀（L-113C 事前規則）\n\n- R1 使 Calmar 由 {B['t']['Calmar']:.2f} 變為 {R1['t']['Calmar']:.2f}（下降 {drop:.2f}；門檻 0.165）→ "
        f"{'**興櫃為主因**' if main_cause else '興櫃**不是**主因'}。")
    log(f"- 剩餘差距（R1 − 新基準 0.89）：{R1['t']['Calmar'] - 0.89:+.2f}；R2 − 新基準：{R2['t']['Calmar'] - 0.89:+.2f}。")
    head = f"興櫃{'為' if main_cause else '不是'}主因：剔除後 2005–15 Calmar {B['t']['Calmar']:.2f}→{R1['t']['Calmar']:.2f}（新基準 0.89）"
    return dict(summary='# L-113C 舊資料中的興櫃價格\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=0, series={})
