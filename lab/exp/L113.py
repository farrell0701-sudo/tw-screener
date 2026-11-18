# L-113 驗收（步驟 6）：依 registry_L.md L-113 及其補充執行驗收 A–D。不是新策略測試，試驗次數不累加（n_variants＝0）。
# 執行：python lab/core/run.py L-113
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100
TARGET = {'2005–15': (0.280, 1.16), '2016–20': (0.252, 2.81), '2021–26': (0.349, 3.24)}   # 舊引擎（年化、Calmar）
L112 = {'2008/07–15': 1.18, '2016–20': 1.97, '2021–23': 3.97}
SEG3 = {k: E.SEGS[k] for k in ('2005–15', '2016–20', '2021–26', '2016–26', '全期')}
SEGC = {'2008/07–15': ('2008-07-01', '2015-12-31'), '2016–20': ('2016-01-01', '2020-12-31'), '2021–23': ('2021-01-01', '2023-12-31'),
        '全期': ('2008-07-01', '2023-12-31')}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)

P = load_panel(); dates = P['dates']; codes = P['meta']['security_key'].to_numpy()
FC = {'AF_C': load_feat('AF_C'), 'AF': load_feat('AF')}
MD = signals.MODES


def mode_dict(base, **kw): return dict(MD[base], **kw)


def build_sig(name, mode, **sw):
    md = MD[mode] if isinstance(mode, str) else mode
    signals.build(name, mode=md, P=P, F=FC['AF_C' if md['bias2'] else 'AF'], **sw)


def run(mode, sig, start=None, end_date=None, x_kw=None, v_kw=None):
    t0 = time.time()
    xk = dict(x_kw or {}); vk = dict(v_kw or {})
    if end_date is not None:
        e = int(dates.searchsorted(pd.Timestamp(end_date))); xk['end'] = e; vk['end'] = e
    R = sim.run_paths(mode, NPATH, xk, vk, sig=sig, start=start)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    log(f'  （{sig}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M)


def table(out, segs=SEG3, which='M'):
    paths = out['M'] if which == 'M' else [x[0] for x in out['R'][which]]
    return E.seg_table(paths, segs)


def md_table(df, cols=('年化', '回撤', 'Calmar', 'Sharpe', 'Calmar_P10', '最差回撤')):
    t = E.fmt_table(df[list(cols)])
    lines = ['| 期間 | ' + ' | '.join(t.columns) + ' |', '|---' * (len(t.columns) + 1) + '|']
    lines += ['| ' + str(i) + ' | ' + ' | '.join(t.loc[i]) + ' |' for i in t.index]
    return '\n'.join(lines)


def top5(out):
    rows = []
    for k, w in (('X', 0.6), ('V', 0.4)):
        for i in range(NPATH):
            for tr in out['R'][k][i][1]: rows.append((tr[0], tr[3] * tr[4] * w / NPATH))
    s = pd.DataFrame(rows, columns=['code', 'c']).groupby('code')['c'].sum().sort_values(ascending=False)
    top = list(s.index[:5]); return top, np.where(np.isin(codes, top))[0]


def taiex_calmar(segs):
    r = P['index']['TAIEX_P'].pct_change(fill_method=None)
    return {k: E.met(r.loc[lo:hi])[2] for k, (lo, hi) in segs.items()}


def main():
    S = {}; series = {}
    # ================= A. 相容模式重現定版數字，並逐一關掉偏誤拆解
    log('## A. 相容模式重現定版數字（小資金合併版 1 倍，100 路徑中位數）\n')
    build_sig('compat', 'compat'); S['compat'] = run('compat', 'compat'); A = table(S['compat'])
    log(md_table(A)); log('')
    rows = []; okA = True
    for k, (ann, cal) in TARGET.items():
        da, dc = A.loc[k, '年化'] - ann, A.loc[k, 'Calmar'] - cal
        ok = abs(da) <= 0.03 and abs(dc) <= 0.25; okA &= ok
        rows.append(f"| {k} | {ann:.1%} / {cal:.2f} | {A.loc[k, '年化']:.1%} / {A.loc[k, 'Calmar']:.2f} | {da * 100:+.1f}pt / {dc:+.2f} | {'通過' if ok else '**不通過**'} |")
    log('| 期間 | 舊引擎 年化 / Calmar | 新引擎相容模式 | 差 | 判定（±3pt、±0.25） |\n|---|---|---|---|---|\n' + '\n'.join(rows))
    log(f"\n**驗收 A：{'通過' if okA else '不通過'}**\n")
    log('### A2. 偏誤拆解（合併版 Calmar；每列只改一項）\n')
    decomp = [('相容（三個偏誤全開）', 'compat', 'compat', MD['compat']),
              ('關掉 ①（納入 2016 起下市股）', 'c_nb1', None, mode_dict('compat', bias1=False)),
              ('關掉 ②（2016/7 前股利還原）', 'c_nb2', None, mode_dict('compat', bias2=False)),
              ('關掉 ③（停牌／下市照修正規則）', 'c_nb3', None, mode_dict('compat', bias3=False)),
              ('修正模式（全關＋成交金額＋主檔產業）', 'fix', None, MD['fix'])]
    lines = ['| 版本 | 2005–15 | 2016–20 | 2021–26 | 2016–26 | 全期回撤 |', '|---|---|---|---|---|---|']
    for nm, sig, _, md in decomp:
        if sig != 'compat':
            build_sig(sig, md); S[sig] = run(md, sig)
        t = table(S[sig])
        lines.append(f"| {nm} | {t.loc['2005–15', 'Calmar']:.2f}（{t.loc['2005–15', '年化']:.1%}） | {t.loc['2016–20', 'Calmar']:.2f}（{t.loc['2016–20', '年化']:.1%}） | "
                     f"{t.loc['2021–26', 'Calmar']:.2f}（{t.loc['2021–26', '年化']:.1%}） | {t.loc['2016–26', 'Calmar']:.2f} | {t.loc['全期', '回撤']:.1%} |")
    log('\n'.join(lines)); log('\n括號內為年化。\n')

    # ================= B. 結論方向（相容模式）
    log('## B. 結論方向（相容模式，合併版）\n')
    var = {'NONE': dict(x_hot=False, x_steep=False), 'HOT': dict(x_steep=False), 'SLOPE': dict(x_hot=False),
           'NODEEP': dict(v_deep=False), 'NOIND': dict(v_ind=False)}
    for k, sw in var.items():
        build_sig(f'c_{k}', 'compat', **sw); S[f'c_{k}'] = run('compat', f'c_{k}')
    cal = lambda key, seg='2016–26': table(S[key]).loc[seg, 'Calmar']
    pm = lambda key: E.path_mean(S[key]['M'])
    lines = ['| 版本 | 2005–15 | 2016–20 | 2021–26 | 2016–26 |', '|---|---|---|---|---|']
    for nm, key in (('BOTH（定版）', 'compat'), ('HOT（只剔過熱）', 'c_HOT'), ('SLOPE（只剔最陡）', 'c_SLOPE'), ('NONE（都不剔）', 'c_NONE'),
                    ('無剔深回撤（L-096 前）', 'c_NODEEP'), ('無產業過濾（L-106 V1）', 'c_NOIND')):
        t = table(S[key]); lines.append(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}" for s in ('2005–15', '2016–20', '2021–26', '2016–26')) + ' |')
    log('\n'.join(lines)); log('')
    b = {}
    c4 = {k: cal(v) for k, v in (('BOTH', 'compat'), ('HOT', 'c_HOT'), ('SLOPE', 'c_SLOPE'), ('NONE', 'c_NONE'))}
    b['L-101 排序 BOTH > HOT > SLOPE > NONE（2016–26）'] = c4['BOTH'] > c4['HOT'] > c4['SLOPE'] > c4['NONE']
    p96 = E.boot_p(pm('compat'), pm('c_NODEEP')); p100 = E.boot_p(pm('compat'), pm('c_HOT'))
    b[f"L-096 剔深回撤仍為改善（2016–26 Calmar {cal('c_NODEEP'):.2f} → {cal('compat'):.2f}，P {p96:.2f}）"] = cal('compat') > cal('c_NODEEP')
    b[f"L-100 剔最陡仍為改善（2016–26 Calmar {cal('c_HOT'):.2f} → {cal('compat'):.2f}，P {p100:.2f}）"] = cal('compat') > cal('c_HOT')
    b[f"L-106 V1 拿掉產業過濾仍變差（2016–26 Calmar {cal('compat'):.2f} → {cal('c_NOIND'):.2f}）"] = cal('c_NOIND') < cal('compat')
    for k, v in b.items(): log(f"- {'通過' if v else '**不通過**'}：{k}")
    okB = all(b.values()); log(f"\n**驗收 B：{'通過' if okB else '不通過'}**（判準只看方向；P 值僅供參考）\n")

    # ================= C. FinLab 方向比對（2008/07–2023/12）
    log('## C. FinLab 方向比對（2008/07–2023/12，對照 L-112）\n')
    tx = taiex_calmar(SEGC); okC = {}
    for mode in ('compat', 'fix'):
        o = run(mode, mode, start='2008-07-01', end_date='2024-01-01'); t = table(o, SEGC)
        top, ban = top5(o)
        ob = run(mode, mode, start='2008-07-01', end_date='2024-01-01', x_kw={'ban': ban}, v_kw={'ban': ban}); tb = table(ob, SEGC)
        tX = table(o, SEGC, 'X'); tV = table(o, SEGC, 'V')
        segs = ['2008/07–15', '2016–20', '2021–23']
        c1 = all(t.loc[s, 'Calmar'] > 1 and t.loc[s, 'Calmar'] > tx[s] for s in segs)
        c2 = t.loc[segs[0], 'Calmar'] < t.loc[segs[1], 'Calmar'] < t.loc[segs[2], 'Calmar']
        c3 = all(tb.loc[s, 'Calmar'] > 0.8 for s in segs)
        okC[mode] = c1 and c2 and c3
        log(f'### {"相容模式" if mode == "compat" else "修正模式"}\n')
        log('| 期間 | 合併 | 剔除前 5 名 | X1 單獨 | V4.1 單獨 | 加權指數（價格） | L-112 FinLab 合併 |\n|---|---|---|---|---|---|---|')
        for s in segs + ['全期']:
            log(f"| {s} | {t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%} / {t.loc[s, '回撤']:.1%}） | {tb.loc[s, 'Calmar']:.2f} | {tX.loc[s, 'Calmar']:.2f} | "
                f"{tV.loc[s, 'Calmar']:.2f} | {tx[s]:.2f} | {L112.get(s, 0.92 if s == '全期' else np.nan):.2f} |")
        log(f"\n- 判準 1（三段 Calmar > 1 且高於指數）：{'成立' if c1 else '**不成立**'}")
        log(f"- 判準 2（三段順序同 L-112：2008–15 < 2016–20 < 2021–23）：{'成立' if c2 else '**不成立**'}")
        log(f"- 判準 3（剔除前 5 名後三段 Calmar > 0.8）：{'成立' if c3 else '**不成立**'}")
        log(f"- 只記錄：2008–15 X1 單獨 {'優於' if tX.loc[segs[0], 'Calmar'] > t.loc[segs[0], 'Calmar'] else '不優於'}合併（L-112 為優於）\n")
    log(f"**驗收 C（相容模式）：{'通過' if okC['compat'] else '不通過'}**；修正模式：{'方向一致' if okC['fix'] else '方向不一致'}\n")

    # ================= D. 額外檢查
    log('## D. 額外檢查\n')
    log('### D1. 剔除報酬貢獻前 5 名股票（全期 2005 起）\n')
    lines = ['| 模式 | 版本 | 2005–15 | 2016–20 | 2021–26 | 全期回撤 |', '|---|---|---|---|---|---|']
    for mode in ('compat', 'fix'):
        top, ban = top5(S[mode]); ob = run(mode, mode, x_kw={'ban': ban}, v_kw={'ban': ban})
        for nm, o in (('全部', S[mode]), ('剔除前 5 名', ob)):
            t = table(o); lines.append(f"| {mode} | {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}" for s in ('2005–15', '2016–20', '2021–26')) + f" | {t.loc['全期', '回撤']:.1%} |")
    log('\n'.join(lines)); log('')

    log('### D2. 下市處理：主版本 A vs 必報檢查 B（修正模式，L-113 第 3 點修訂）\n')
    lk = [sum(1 for x in S['fix']['R'][k][i][1] if x[8]) for k in ('X', 'V') for i in range(NPATH)]
    dl = [sum(1 for x in S['fix']['R'][k][i][1] if x[6] == '下市出清') for k in ('X', 'V') for i in range(NPATH)]
    log(f'- 每條路徑平均：下市出清 {np.mean(dl):.2f} 筆，其中最後交易日跌停鎖死 {np.mean(lk):.2f} 筆（X1＋V4.1）。')
    tA = table(S['fix']); lines = ['| 版本 | 2005–15 | 2016–20 | 2021–26 | 2016–26 |', '|---|---|---|---|---|',
                                   '| A（最後交易日收盤出清） | ' + ' | '.join(f"{tA.loc[s, 'Calmar']:.2f}" for s in ('2005–15', '2016–20', '2021–26', '2016–26')) + ' |']
    worst = 0.0
    for disc in (0.0, 0.5, 1.0):
        o = run('fix', 'fix', x_kw={'delist': ('B', disc)}, v_kw={'delist': ('B', disc)}); t = table(o)
        lines.append(f'| B（鎖死者凍結至下市日，折扣 {disc:.0%}） | ' + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}" for s in ('2005–15', '2016–20', '2021–26', '2016–26')) + ' |')
        if disc == 1.0: worst = max(abs(t.loc[s, 'Calmar'] - tA.loc[s, 'Calmar']) for s in ('2005–15', '2016–20', '2021–26'))
    log('\n'.join(lines))
    okD2 = worst <= 0.1
    log(f"\n- 事前判準：任一段 A 與 B（100% 折扣）Calmar 差 > 0.1 → 回報使用者重新決定。最大差 {worst:.2f} → {'維持 A' if okD2 else '**需回報使用者**'}\n")

    log('### D3. 停牌與下市部位（100 路徑合計）\n')
    lines = ['| 模式 | 策略 | 交易筆數 | 曾連續 20 日無有效價格 | 下市出清 | 250 日上限 | 期末未平倉 |', '|---|---|---|---|---|---|---|']
    for mode in ('compat', 'fix'):
        for k in ('X', 'V'):
            trs = [x for i in range(NPATH) for x in S[mode]['R'][k][i][1]]
            n = len(trs); f = lambda c: sum(1 for x in trs if c(x))
            lines.append(f"| {mode} | {'X1' if k == 'X' else 'V4.1'} | {n} | {f(lambda x: x[7])}（{f(lambda x: x[7]) / n:.1%}） | {f(lambda x: x[6] == '下市出清')} | "
                         f"{f(lambda x: x[6] == '250日上限')} | {f(lambda x: x[6] == '期末未平倉')} |")
    log('\n'.join(lines)); log('')

    log('## 新基準（修正模式，小資金合併版 1 倍）\n'); log(md_table(table(S['fix']))); log('')
    for k, nm in (('X', 'X1 單獨'), ('V', 'V4.1 單獨')):
        log(f'### {nm}（修正模式）\n'); log(md_table(table(S['fix'], which=k))); log('')

    for mode in ('compat', 'fix'):
        series[f'{mode}_merge'] = E.path_mean(S[mode]['M'])
        series[f'{mode}_X'] = E.path_mean([x[0] for x in S[mode]['R']['X']]); series[f'{mode}_V'] = E.path_mean([x[0] for x in S[mode]['R']['V']])
    head = f"驗收 A {'通過' if okA else '不通過'}、B {'通過' if okB else '不通過'}、C {'通過' if okC['compat'] else '不通過'}、下市 B 檢查{'維持 A' if okD2 else '需回報'}"
    summary = f'# L-113 驗收結果\n\n**{head}**\n\n' + '\n'.join(LOG)
    return dict(summary=summary, headline=head, series=series, n_variants=0)
