# M-184：R6 頭部剔除月初弱勢的交易報酬分布：15% 硬停損 vs 不停損（奇數代碼，描述）。定義見 registry_M.md M-184。
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M182

ERAS = M182.ERAS
BLUE, ORANGE, INK, INK2 = '#257de0', '#f47230', '#191919', '#666666'


def cjk_font():
    for f in sorted(font_manager.findSystemFonts(), key=lambda p: ('regular' not in p.lower(), 'serif' in p.lower())):
        n = f.lower()
        if any(k in n for k in ('notosanscjk', 'noto sans cjk', 'wqy', 'droidsansfallback', 'notosanstc', 'sourcehansans')):
            font_manager.fontManager.addfont(f); return font_manager.FontProperties(fname=f).get_name()
    return None


def stats(x):
    x = x.dropna()
    return dict(n=len(x), mean=x.mean(), med=x.median(), win=(x > 0).mean(), p5=x.quantile(.05), p25=x.quantile(.25),
                p75=x.quantile(.75), p95=x.quantile(.95), gt2=(x > 0.02).mean())


def main():
    D = M182.build(parity=1); D = D[D.q >= 2]
    fn = cjk_font()
    if fn: plt.rcParams['font.family'] = fn
    plt.rcParams['axes.unicode_minus'] = False
    L = ['# M-184 結果：R6 頭部剔除月初弱勢｜交易報酬分布：15% 硬停損 vs 不停損（奇數代碼）\n',
         '進場 t+1 開盤；S15＝收盤跌破進場 ×0.85 次日開盤出場、最長 120 日；NS＝不停損持有；皆扣來回 0.77%。\n']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    bins = np.arange(-0.60, 1.21, 0.025)
    for ax, (era, lo, hi) in zip(axes, ERAS):
        E = D[(D.date >= lo) & (D.date <= hi)]
        L += [f'## {era}（{len(E):,} 筆）\n', '| 版本 | 平均 | 中位數 | 勝率 | > +2% 比例 | P5 | P25 | P75 | P95 |', '|---|---|---|---|---|---|---|---|---|']
        for nm, col in (('S15 15% 硬停損（≤120 日）', 'net'), ('NS120 不停損 120 日', 'net_ns'), ('NS60 不停損 60 日', 'net_ns60')):
            s = stats(E[col])
            L.append(f"| {nm} | {s['mean']:+.2%} | {s['med']:+.2%} | {s['win']:.1%} | {s['gt2']:.1%} | {s['p5']:+.1%} | {s['p25']:+.1%} | {s['p75']:+.1%} | {s['p95']:+.1%} |")
        st = E[E.stop]
        rec = st.net_ns
        L += ['', f"被停損 {len(st):,} 筆（{E.stop.mean():.0%}）：若不停損，120 日後轉正 {(rec > 0).mean():.0%}、> +2% {(rec > 0.02).mean():.0%}、中位數 {rec.median():+.1%}、"
              f"跌超過 −30% {(rec < -0.30).mean():.0%}；停損出場本身中位數 {st.net.median():+.1%}。",
              f"未被停損 {len(E) - len(st):,} 筆：勝率 {(E[~E.stop].net > 0).mean():.0%}、中位數 {E[~E.stop].net.median():+.1%}。\n"]
        a = E.net.clip(bins[0], bins[-1]); b = E.net_ns.clip(bins[0], bins[-1])
        ax.hist(b, bins=bins, weights=np.full(len(b), 100 / len(b)), color=BLUE, alpha=0.55, label='不停損（120 日）', edgecolor='white', linewidth=0.5)
        ax.hist(a, bins=bins, weights=np.full(len(a), 100 / len(a)), histtype='step', color=ORANGE, linewidth=2, label='15% 硬停損')
        for v, c, ls in ((E.net_ns.median(), BLUE, '--'), (E.net.median(), ORANGE, '--')):
            ax.axvline(v, color=c, linestyle=ls, linewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{era}　勝率：停損 {(E.net > 0).mean():.0%}｜不停損 {(E.net_ns > 0).mean():.0%}", color=INK, fontsize=11)
        ax.set_xlabel('每筆淨報酬（兩端截在 −60%／+120%；虛線＝中位數）', color=INK2, fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis='y', color='#e6e6e6', linewidth=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].set_ylabel('占交易筆數 %', color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('R6 頭部剔除月初弱勢：每筆報酬分布（奇數代碼）', color=INK, fontsize=12)
    fig.tight_layout()
    out = LAB / 'results' / 'M184'; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'dist.png', dpi=150, facecolor='white')
    L.append('![分布](dist.png)\n')
    return dict(summary='\n'.join(L), headline='停損 vs 不停損分布', n_variants=0)
