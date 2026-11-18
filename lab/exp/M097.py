# M-097：X1 與 V4.1 在合併版的邊際效果（描述性，新基準＝修正模式）。不作採用判準；權重曲線只看 60/40 是否在高原。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100
SEG = {k: E.SEGS[k] for k in ('2005–15', '2016–20', '2021–26', '2016–26')}
SEG3 = {k: E.SEGS[k] for k in ('2005–15', '2016–20', '2021–26')}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def dd_episodes(r, k=10):
    e = (1 + r).cumprod(); dd = e / e.cummax() - 1; out = []; i = 0; v = dd.values; idx = dd.index
    while i < len(v):
        if v[i] < 0:
            j = i
            while j < len(v) and v[j] < 0: j += 1
            seg = dd.iloc[i:j]; out.append((idx[i - 1] if i > 0 else idx[0], seg.idxmin(), idx[j] if j < len(v) else idx[-1], seg.min())); i = j
        else: i += 1
    return sorted(out, key=lambda x: x[3])[:k]


def main():
    P = load_panel(); F = load_feat('AF')
    S = signals.compute('fix', P=P, F=F)
    R = sim.run_paths('fix', NPATH)
    X = [R['X'][i][0] for i in range(NPATH)]; V = [R['V'][i][0] for i in range(NPATH)]
    xm, vm = E.path_mean(X), E.path_mean(V); xm, vm = xm.align(vm, fill_value=0)
    M = [sim.mix(X[i], V[i]) for i in range(NPATH)]; mm = E.path_mean(M).reindex(xm.index).fillna(0)
    below = S['daily']['below'].reindex(xm.index).fillna(False).astype(bool)

    # 1. 相關
    log('## 1. X1 與 V4.1 報酬相關（100 路徑平均報酬）\n')
    log('| 期間 | 日相關 | 月相關 | 合併回撤 > 5% 期間（日） | 指數 < 200MA 期間（日） |\n|---|---|---|---|---|')
    e = (1 + mm).cumprod(); ddm = e / e.cummax() - 1
    mon = lambda r: (1 + r).groupby(r.index.to_period('M')).prod() - 1
    for nm, (lo, hi) in {**SEG3, '全期': ('2005-01-01', '2026-12-31')}.items():
        x, v = xm.loc[lo:hi], vm.loc[lo:hi]; d_ = ddm.loc[lo:hi] < -0.05; b_ = below.loc[lo:hi]
        log(f"| {nm} | {x.corr(v):.2f} | {mon(x).corr(mon(v)):.2f} | {x[d_].corr(v[d_]):.2f}（{d_.mean():.0%} 的日子） | {x[b_].corr(v[b_]):.2f}（{b_.mean():.0%}） |")

    # 2. 回撤重疊
    log('\n## 2. 合併版前 10 大回撤：同期 X1、V4.1 的報酬與各自最大回撤\n')
    log('| 高點 | 谷底 | 合併 | X1 同期報酬 | X1 期間內最大回撤 | V4.1 同期報酬 | V4.1 期間內最大回撤 |\n|---|---|---|---|---|---|---|')
    for pk, tr, rc, depth in dd_episodes(mm):
        def st(r):
            s_ = r.loc[pk:tr].iloc[1:]; e_ = (1 + s_).cumprod(); return e_.iloc[-1] - 1, (e_ / e_.cummax().clip(lower=1) - 1).min()
        xr, xd = st(xm); vr, vd = st(vm)
        log(f"| {pk.date()} | {tr.date()} | {depth:.1%} | {xr:+.1%} | {xd:.1%} | {vr:+.1%} | {vd:.1%} |")

    # 3. 權重曲線
    log('\n## 3. 權重曲線（X1 比例；每月再平衡；Calmar 與年化 / 回撤，100 路徑中位數）\n')
    log('| X1 比例 | ' + ' | '.join(SEG) + ' |\n|---|---|---|---|---|')
    curve = {}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        Mw = X if w == 1 else (V if w == 0 else [sim.mix(X[i], V[i], w) for i in range(NPATH)])
        t = E.seg_table(Mw, SEG); curve[w] = t
        log(f"| {w:.0%}{'（定版）' if w == 0.6 else ''} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%} / {t.loc[s, '回撤']:.1%}）" for s in SEG) + ' |')
    best = {s: max(curve, key=lambda w: curve[w].loc[s, 'Calmar']) for s in SEG}
    log('\n各段 Calmar 最高的 X1 比例：' + '、'.join(f'{s} {best[s]:.0%}' for s in SEG) + '（只作高原確認，不據此調整權重）。')
    plat = {s: [w for w in curve if curve[w].loc[s, 'Calmar'] >= 0.9 * curve[best[s]].loc[s, 'Calmar']] for s in SEG}
    log('Calmar 在最高值 90% 以內的範圍：' + '、'.join(f'{s} {min(v):.0%}–{max(v):.0%}' for s, v in plat.items()) + '。\n')

    # 4. 分年度貢獻
    log('## 4. 分年度報酬（路徑平均；貢獻＝0.6 × X1、0.4 × V4.1，未計再平衡漂移）\n')
    yr = lambda r: (1 + r).groupby(r.index.year).prod() - 1
    Y = pd.DataFrame({'X1': yr(xm), 'V4.1': yr(vm), '合併': yr(mm)}).loc[2005:]
    log('| 年 | X1 | V4.1 | 合併 | X1 貢獻 | V4.1 貢獻 | 較強者 |\n|---|---|---|---|---|---|---|')
    for y, r in Y.iterrows():
        log(f"| {y} | {r['X1']:+.1%} | {r['V4.1']:+.1%} | {r['合併']:+.1%} | {0.6 * r['X1']:+.1%} | {0.4 * r['V4.1']:+.1%} | {'X1' if r['X1'] > r['V4.1'] else 'V4.1'} |")

    # 5. 盤勢：強勢池月換手率三分位
    log('\n## 5. 盤勢代理：強勢池月換手率（三分位）與兩者相對表現\n')
    pool = S['_dbg']['pool']; d = P['dates']; cols = S['_dbg']['cols']
    me = pd.Series(np.arange(len(d)), index=d).groupby(d.to_period('M')).last()
    me = me[me.index >= pd.Period('2004-12', 'M')]
    sets = [set(np.flatnonzero(pool[t])) for t in me.values]
    to = pd.Series([np.nan] + [1 - len(a & b) / max(len(a), 1) for a, b in zip(sets[:-1], sets[1:])], index=me.index)
    # 換手率 m 月（月底 m 對 m−1）→ 對應 m+1 月報酬（事前可知）；另報同月（描述）
    mx, mv = mon(xm), mon(vm)
    df = pd.DataFrame({'to': to, 'x_next': mx.shift(-1).reindex(to.index), 'v_next': mv.shift(-1).reindex(to.index),
                       'x_same': mx.reindex(to.index), 'v_same': mv.reindex(to.index)}).dropna()
    df = df[df.index >= pd.Period('2005-01', 'M')]
    df['q'] = pd.qcut(df['to'], 3, labels=['低', '中', '高'])
    log('| 換手率分組 | 月數 | 換手率範圍 | 同月 X1 | 同月 V4.1 | 同月 V4.1−X1 | 次月 X1 | 次月 V4.1 | 次月 V4.1−X1 |\n|---|---|---|---|---|---|---|---|---|')
    for q, g in df.groupby('q', observed=True):
        log(f"| {q} | {len(g)} | {g.to.min():.0%}–{g.to.max():.0%} | {g.x_same.mean():+.2%} | {g.v_same.mean():+.2%} | {(g.v_same - g.x_same).mean():+.2%} | "
            f"{g.x_next.mean():+.2%} | {g.v_next.mean():+.2%} | {(g.v_next - g.x_next).mean():+.2%} |")
    hi_, lo_ = df[df.q == '高'], df[df.q == '低']
    dlt = (hi_.v_same - hi_.x_same).mean() - (lo_.v_same - lo_.x_same).mean()
    log(f"\n高換手 − 低換手的「V4.1−X1」差（同月）：{dlt:+.2%}／月。")

    # 事前預期對照
    c = {nm: xm.loc[lo:hi].corr(vm.loc[lo:hi]) for nm, (lo, hi) in SEG3.items()}
    e1 = max(c, key=c.get) == '2005–15'
    e2 = all(curve[w].loc[s, 'Calmar'] >= 0.9 * curve[best[s]].loc[s, 'Calmar'] for s in SEG for w in (0.4, 0.5, 0.6, 0.7, 0.8))
    e3 = dlt > 0
    log('\n## 事前預期對照\n')
    log(f"- 相關係數 05-15 最高：{'符合' if e1 else '**不符**'}（日相關 " + '、'.join(f'{k} {v:.2f}' for k, v in c.items()) + '）')
    log(f"- 權重曲線在 40%～80% 平坦（各段 Calmar 都在最高值 90% 以內）：{'符合' if e2 else '**不符**'}")
    log(f"- 換手率高的月份 V4.1 相對 X1 較佳：{'符合' if e3 else '**不符**'}（{dlt:+.2%}／月）")
    head = f"M-097 描述：相關 05-15 最高{'✓' if e1 else '✗'}、權重 40–80% 高原{'✓' if e2 else '✗'}、高換手 V4.1 較佳{'✓' if e3 else '✗'}"
    return dict(summary='# M-097 X1 與 V4.1 的邊際效果（新基準）\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=0,
                series={'X1': xm, 'V41': vm, 'merge': mm})
