# L-113B：驗收 A 差異拆解——舊資料（old-data、history-data）跑新引擎（相容模式＝參考實作）。試驗次數不累加。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
import old_panel, feat, signals, sim, evalu as E

NPATH = 100
MODE = dict(bias1=False, bias2=True, bias3=True, amt='AMT20', ind='pk')     # 相容：成交額＝收盤×量；產業欄位由 old_panel 決定
TARGET = {'2005–15': (0.280, -0.239, 1.16), '2016–20': (0.252, -0.089, 2.81), '2021–26': (0.349, -0.108, 3.24)}
NEW = {'2005–15': (0.229, 0.89), '2016–20': (0.236, 2.53), '2021–26': (0.294, 2.59)}          # L-113 修正模式
COMPAT = {'2005–15': (0.168, 0.68), '2016–20': (0.244, 2.52), '2021–26': (0.354, 3.56)}       # L-113 相容模式
WIN = {'old': ('2005-01-01', {'2005–15': ('2005-01-01', '2015-12-31')}),
       'new': ('2016-02-01', {'2016–20': ('2016-02-01', '2020-12-31'), '2021–26': ('2021-01-01', '2026-12-31')})}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def run_window(which, industry):
    t0 = time.time()
    P = old_panel.build(which, industry); F = feat.compute(P, 'AF_C'); S = signals.compute(MODE, P=P, F=F)
    start, segs = WIN[which]; c = sim.make_ctx(MODE, P, F, S, start=start)
    X = [sim.runX(c, off=i % 20, seed=1000 + i) for i in range(NPATH)]
    V = [sim.runV(c, off=i % 20, seed=1000 + i) for i in range(NPATH)]
    Mx = [sim.mix(x[0], v[0]) for x, v in zip(X, V)]
    d = S['daily']; out = {}
    for nm, (lo, hi) in segs.items():
        out[nm] = dict(M=E.seg_table(Mx, {nm: (lo, hi)}).iloc[0], X=E.seg_table([x[0] for x in X], {nm: (lo, hi)}).iloc[0],
                       V=E.seg_table([v[0] for v in V], {nm: (lo, hi)}).iloc[0],
                       ev=int(d.loc[lo:hi, 'n_signal'].sum()), pool=float(d.loc[lo:hi, 'n_pool'].mean()), U=float(d.loc[lo:hi, 'n_U'].mean()))
    info = dict(n=len(P['meta']), fix=P['_n_fix'], ind=P['_n_ind'], d0=str(P['dates'][0].date()), d1=str(P['dates'][-1].date()), sec=round(time.time() - t0),
                ev_all=(int(d.loc['2005':'2015', 'n_signal'].sum()) if which == 'old' else int(d.loc['2016':, 'n_signal'].sum())))
    return out, info


def main():
    R = {}
    for which in ('old', 'new'):
        for ind in ('list', 'pk'):
            R[(which, ind)] = run_window(which, ind); i = R[(which, ind)][1]
            log(f"- {which}／產業 {ind}：{i['n']} 檔（有產業 {i['ind']}）、{i['d0']}～{i['d1']}、修正跳動 {i['fix']} 筆、{i['sec']} 秒")
    log('\n## 合併版（年化 / 回撤 / Calmar，100 路徑中位數）\n')
    log('| 期間 | 舊引擎 | 舊資料＋新引擎（產業 stock_list） | 舊資料＋新引擎（產業 PK） | L-113 相容（新資料） | L-113 修正 |\n|---|---|---|---|---|---|')
    okA = {}
    for seg in ('2005–15', '2016–20', '2021–26'):
        which = 'old' if seg == '2005–15' else 'new'
        cells = []
        for ind in ('list', 'pk'):
            m = R[(which, ind)][0][seg]['M']; cells.append(f"{m['年化']:.1%} / {m['回撤']:.1%} / {m['Calmar']:.2f}")
            okA[(seg, ind)] = abs(m['年化'] - TARGET[seg][0]) <= 0.03 and abs(m['Calmar'] - TARGET[seg][2]) <= 0.25
        a, dd, cal = TARGET[seg]
        log(f"| {seg} | {a:.1%} / {dd:.1%} / {cal:.2f} | {cells[0]} | {cells[1]} | {COMPAT[seg][0]:.1%} / {COMPAT[seg][1]:.2f} | {NEW[seg][0]:.1%} / {NEW[seg][1]:.2f} |")
    log('\n## 子策略與股池（產業 stock_list）\n')
    log('| 期間 | X1 年化 / Calmar | V4.1 年化 / Calmar | V4.1 事件數 | 強勢池/日 | 可交易池/日 |\n|---|---|---|---|---|---|')
    for seg in ('2005–15', '2016–20', '2021–26'):
        o = R[('old' if seg == '2005–15' else 'new', 'list')][0][seg]
        log(f"| {seg} | {o['X']['年化']:.1%} / {o['X']['Calmar']:.2f} | {o['V']['年化']:.1%} / {o['V']['Calmar']:.2f} | {o['ev']} | {o['pool']:.1f} | {o['U']:.0f} |")
    log(f"\nV4.1 事件數對照：L-106 舊引擎 2,397（05-15）｜2,693（16-26）；本次舊窗 {R[('old', 'list')][1]['ev_all']}、新窗（2016 起）{R[('new', 'list')][1]['ev_all']}；"
        f"L-113 新資料相容 1,750 ｜ 2,369。\n")
    log('## 判讀（依 L-113B 事前規則）\n')
    for seg in ('2005–15', '2016–20', '2021–26'):
        log(f"- {seg}：舊資料版本 {'在' if okA[(seg, 'list')] else '**不在**'}驗收 A 容許範圍內（產業 stock_list）；PK 版本 {'在' if okA[(seg, 'pk')] else '**不在**'}範圍內。")
    ok_all = all(okA[(s, 'list')] for s in ('2005–15', '2016–20', '2021–26'))
    head = '舊資料＋新引擎' + ('重現舊數字：落差來自資料' if ok_all else '仍有段落重現不了：部分差異未解釋')
    return dict(summary='# L-113B 舊資料跑新引擎\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=0,
                series={})
