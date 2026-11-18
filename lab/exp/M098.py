# M-098：剔最陡在不同年代的意義（描述性）。設計見 registry_M.md M-098 與「實作細節」補記。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals

ERAS = {'05-15': ('2005-01-01', '2015-12-31'), '16-20': ('2016-01-01', '2020-12-31'), '21-26': ('2021-01-01', '2026-12-31'),
        '16-26': ('2016-01-01', '2026-12-31')}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def main():
    P = load_panel(); F = load_feat('AF'); d = P['dates']; M = P['meta']
    S = signals.compute('fix', P=P, F=F); D = S['_dbg']
    pool, steep, sig = D['pool'], D['steep'], D['signal']
    S60 = np.asarray(F['S60'], float); VALID = np.asarray(F['VALID'])
    up5 = pool & (np.nan_to_num(S60, nan=-1) >= 0.05)
    below = S['daily']['below'].to_numpy(bool)
    Cf = pd.DataFrame(np.asarray(F['CA'], float)).ffill().values
    FW = {}
    for h in (20, 60, 120):
        f = np.full(Cf.shape, np.nan); f[:-h] = Cf[h:] / Cf[:-h] - 1; f[~VALID] = np.nan; FW[h] = f
    last = M['stock_id'].str[-1]
    SETS = {'確認集': last.isin(list('02468')).to_numpy(), '探索集': last.isin(list('13579')).to_numpy()}
    wmask = lambda lo, hi: (d >= lo) & (d <= hi)

    # 1. 斜率 ≥ 5% 股數
    log('## 1. 強勢池中 EMA200 60 日斜率 ≥ 5% 的檔數（每日平均，全部股票）\n')
    log('| 年代 | 盤勢 | 強勢池 | 斜率 ≥ 5% | 占強勢池 | 實際被剔（每日） |\n|---|---|---|---|---|---|')
    n_up5 = {}
    for e, (lo, hi) in list(ERAS.items())[:3]:
        for nm, m in (('全部', np.ones(len(d), bool)), ('多頭', ~below), ('空頭', below)):
            w = wmask(lo, hi) & m
            a, b, c = pool[w].sum(1).mean(), up5[w].sum(1).mean(), (pool & steep)[w].sum(1).mean()
            if nm == '全部': n_up5[e] = b
            log(f'| {e} | {nm} | {a:.1f} | {b:.1f} | {b / a:.1%} | {c:.1f} |')

    # 2. 被剔除者絕對斜率
    log('\n## 2. 被剔除者的 S60 分佈（全部股票）\n')
    log('| 年代 | 中位數 | P25 | P75 | 筆數 |\n|---|---|---|---|---|')
    med = {}
    for e, (lo, hi) in ERAS.items():
        v = S60[wmask(lo, hi)][(pool & steep)[wmask(lo, hi)]]; med[e] = np.median(v)
        log(f'| {e} | {np.median(v):.1%} | {np.percentile(v, 25):.1%} | {np.percentile(v, 75):.1%} | {len(v):,} |')

    # 3. 後續報酬
    def stats(A, B, lo, hi, cm, h):
        w = wmask(lo, hi); f = FW[h][w][:, cm]; a = f[A[w][:, cm]]; b = f[B[w][:, cm]]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        g = lambda x: (x.mean(), np.median(x), np.percentile(x, 90), (x > 0).mean(), len(x)) if len(x) else (np.nan,) * 4 + (0,)
        return g(a), g(b)
    GROUPS = {'X1 強勢池': (pool & steep, pool & ~steep), 'V4.1 事件': (sig & steep, sig & ~steep)}
    R = {}
    for sn, cm in SETS.items():
        log(f'\n## 3. 被剔除（最陡）vs 其餘：後續報酬｜{sn}\n')
        log('| 組 | 年代 | 天數 | 被剔：平均 / 中位 / P90 / 勝率（筆） | 其餘：平均 / 中位 / P90 / 勝率（筆） | 平均差 |\n|---|---|---|---|---|---|')
        for gn, (A, B) in GROUPS.items():
            for e in ('05-15', '16-20', '21-26', '16-26'):
                for h in (20, 60, 120):
                    a, b = stats(A, B, *ERAS[e], cm, h); R[(sn, gn, e, h)] = (a, b)
                    f = lambda x: f'{x[0]:+.1%} / {x[1]:+.1%} / {x[2]:+.1%} / {x[3]:.0%}（{x[4]:,}）'
                    log(f'| {gn} | {e} | {h} | {f(a)} | {f(b)} | {(a[0] - b[0]) * 100:+.1f}pt |')

    # 4. 變體：相對門檻、絕對門檻（被剔比例與現行相同）
    share = (pool & steep).sum() / pool.sum()
    pm = pd.DataFrame(np.where(pool, S60, np.nan)).median(axis=1).to_numpy()
    ratio = np.where(pool & (pm[:, None] > 0), S60 / pm[:, None], np.nan)
    k = np.nanquantile(ratio, 1 - share); x = np.nanquantile(np.where(pool, S60, np.nan), 1 - share)
    V1 = pool & (np.nan_to_num(ratio, nan=-1) >= k); V2 = pool & (np.nan_to_num(S60, nan=-1) >= x)
    log(f'\n## 4. 變體（只描述）：被剔比例與現行相同（{share:.1%}）\n')
    log(f'相對門檻 k＝{k:.2f} 倍強勢池中位數斜率（中位數 ≤ 0 的日子不剔）；絕對門檻 x＝{x:.1%}。實際被剔比例：現行 {share:.1%}、相對 {V1.sum() / pool.sum():.1%}、絕對 {V2.sum() / pool.sum():.1%}。\n')
    log('| 定義 | 集合 | 05-15 被剔 − 其餘（20 日平均） | 16-26 | 兩年代之差 |\n|---|---|---|---|---|')
    gap = {}
    for nm, A in (('現行（百分位）', pool & steep), ('相對門檻', V1), ('絕對門檻', V2)):
        for sn, cm in SETS.items():
            a1, b1 = stats(A, pool & ~A, *ERAS['05-15'], cm, 20); a2, b2 = stats(A, pool & ~A, *ERAS['16-26'], cm, 20)
            d1, d2 = a1[0] - b1[0], a2[0] - b2[0]; gap[(nm, sn)] = abs(d1 - d2)
            log(f'| {nm} | {sn} | {d1 * 100:+.2f}pt | {d2 * 100:+.2f}pt | {abs(d1 - d2) * 100:.2f}pt |')

    # 預期對照（確認集）
    log('\n## 事前預期對照（確認集）\n')
    low = med['05-15'] < 0.8 * med['16-26']; few = n_up5['05-15'] < 10
    ea = not low and not few
    log(f"- (a) 05-15 被剔除者斜率中位數 {med['05-15']:.1%} vs 16-26 {med['16-26']:.1%}（明顯偏低＝< 0.8 倍：{'是' if low else '否'}）；05-15 斜率 ≥ 5% 每日 {n_up5['05-15']:.1f} 檔（很少＝< 10：{'是' if few else '否'}）→ (a) {'成立' if ea else '**不成立**'}；"
        f"斜率 ≥ 5% 股數 16-26 高於 05-15：{'是' if (n_up5['16-20'] + n_up5['21-26']) / 2 > n_up5['05-15'] else '否'}")
    eb = all(R[('確認集', g, '05-15', 20)][0][2] > R[('確認集', g, '05-15', 20)][1][2] and R[('確認集', g, '16-26', 20)][0][2] < R[('確認集', g, '16-26', 20)][1][2] for g in GROUPS)
    log(f"- (b) 20 日 P90：05-15 被剔 > 其餘、16-26 被剔 < 其餘（X1 與 V4.1 皆需）→ {'成立' if eb else '**不成立**'}")
    dd = lambda g, h: (R[('確認集', g, '05-15', h)][0][0] - R[('確認集', g, '05-15', h)][1][0]) - (R[('確認集', g, '16-26', h)][0][0] - R[('確認集', g, '16-26', h)][1][0])
    ec = abs(dd('V4.1 事件', 20)) > abs(dd('X1 強勢池', 120))
    log(f"- (c) 年代差異 V4.1 20 日 {abs(dd('V4.1 事件', 20)) * 100:.1f}pt vs X1 120 日 {abs(dd('X1 強勢池', 120)) * 100:.1f}pt → {'成立' if ec else '**不成立**'}")
    log(f"- 使用者假說：(a) {'成立 → 使用者假說可能性未提高' if ea else '不成立 → 依登記，使用者假說可能性提高'}。")
    log(f"- 變體：兩年代差距（確認集）現行 {gap[('現行（百分位）', '確認集')] * 100:.2f}pt、相對 {gap[('相對門檻', '確認集')] * 100:.2f}pt、絕對 {gap[('絕對門檻', '確認集')] * 100:.2f}pt。")
    head = f"M-098 描述：預期 (a){'✓' if ea else '✗'} (b){'✓' if eb else '✗'} (c){'✓' if ec else '✗'}"
    return dict(summary='# M-098 剔最陡的年代差異\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=0, series={})
