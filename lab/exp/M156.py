# M-156：V4.1 事件剔除 B6 動能品質的 Q1 與 Q5（偶數代碼確認集；只測一次）。定義與判準見 registry_M.md M-156。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41ev import events, removal, stat

LO, HI = 0.2953, 0.7070          # 探索集 B6 20／80 百分位（M-154）
QE = [0.2953, 0.4245, 0.5608, 0.7070]


def main():
    X = events()
    X['per'] = np.where(X.year <= 2015, '2005–15', '2016–26')
    ex = X[(X.par == 1) & (X.year <= 2015)]
    s6 = np.sort(ex.A6.to_numpy()); s1 = np.sort(ex.A1.to_numpy())
    cdf = lambda s, x: np.searchsorted(s, x, side='right') / len(s)
    X['B6'] = 0.5 * cdf(s6, X.A6.to_numpy()) + 0.5 * (1 - cdf(s1, X.A1.to_numpy()))
    eb = X.loc[ex.index, 'B6']
    chk = dict(n=len(ex), p20=float(np.percentile(eb, 20)), p80=float(np.percentile(eb, 80)))
    pl = {'15／85': (float(np.percentile(eb, 15)), float(np.percentile(eb, 85))), '25／75': (float(np.percentile(eb, 25)), float(np.percentile(eb, 75)))}
    PER = ['2005–15', '2016–26']

    pc = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:+.2%}'
    pp = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:.0%}'
    yn = lambda b: '✓' if b else '✗'
    L = ['# M-156 結果：V4.1 事件剔除 B6 Q1＋Q5（確認集：偶數代碼）\n',
         f"重建檢查：探索集事件 {chk['n']}（M-154 為 937）；B6 20／80 百分位 {chk['p20']:.4f}／{chk['p80']:.4f}（登記 {LO}／{HI}）。高原切點 15／85＝{pl['15／85'][0]:.4f}／{pl['15／85'][1]:.4f}、25／75＝{pl['25／75'][0]:.4f}／{pl['25／75'][1]:.4f}。\n"]
    head = ''
    for par_, lab in ((0, '偶數代碼 2005–26（確認集，判準）'), (1, '奇數代碼 2016–26（描述）')):
        D = X[X.par == 0] if par_ == 0 else X[(X.par == 1) & (X.year >= 2016)]
        per = PER if par_ == 0 else ['2016–26']
        L.append(f"## {lab}\n\n事件 {len(D)}" + (f"（2005–15 {int((D.per == '2005–15').sum())}、2016–26 {int((D.per == '2016–26').sum())}）" if par_ == 0 else '') + '\n')
        L += ['| 版本 | 剔除比例 | 被剔 n | 被剔平均 Y | 保留平均 Y | ' + ' | '.join(f'{p} 被剔／保留' for p in per) + ' | 被剔右尾 | 保留右尾 | 左尾占比（vs 剔除比例） | 被剔含停損 | 保留含停損 | (a)(b)(c)(d) |',
              '|---|---|---|---|---|' + '---|' * len(per) + '---|---|---|---|---|---|']
        vers = [('主：剔 Q1＋Q5', (D.B6 < LO) | (D.B6 > HI)), ('高原 15／85', (D.B6 < pl['15／85'][0]) | (D.B6 > pl['15／85'][1])),
                ('高原 25／75', (D.B6 < pl['25／75'][0]) | (D.B6 > pl['25／75'][1])), ('並報：只剔 Q1', D.B6 < LO), ('並報：只剔 Q5', D.B6 > HI)]
        R = {}
        for nm, rm in vers:
            z = removal(D, rm, 'per', per); R[nm] = z; o = z['ok']
            L.append(f"| {nm} | {z['frac']:.0%} | {z['rm']['n']} | {pc(z['rm']['mean'])} | {pc(z['kp']['mean'])} | " + ' | '.join(f"{pc(z['per'][p][0])}／{pc(z['per'][p][1])}" for p in per)
                     + f" | {pp(z['rm']['rt'])} | {pp(z['kp']['rt'])} | {pp(z['lt_share'])}（{pp(z['frac'])}） | {pc(z['rm']['stop'])} | {pc(z['kp']['stop'])} | {yn(o['a'])}{yn(o['b'])}{yn(o['c'])}{yn(o['d'])} |")
        if par_ == 0:
            okv = R['主：剔 Q1＋Q5']['all'] and R['高原 15／85']['ok']['a'] and R['高原 25／75']['ok']['a']
            L.append(f"\n判準（主版本 (a)–(d) 全過，且兩個高原版本 (a) 皆過）：{yn(okv)}\n\n**結論：{'確認' if okv else '未確認，此方向結案'}**\n")
            head = 'B6 剔除 Q1＋Q5 確認集：' + ('確認' if okv else '未確認，結案')
        q = np.digitize(D.B6, QE)
        L += ['五分組（探索集切點）\n', '| 組 | n | 平均 Y | 中位數 | 勝率 | 成功率 | 假突破率 | 左尾 | 右尾 | MAE 中位數 | 含停損 |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for g in range(5):
            s = stat(D[q == g])
            L.append(f"| Q{g + 1} | {s['n']} | {pc(s['mean'])} | {pc(s['med'])} | {pp(s['win'])} | {pp(s['succ'])} | {pp(s['fake'])} | {pp(s['lt'])} | {pp(s['rt'])} | {pc(s['mae'])} | {pc(s['stop'])} |")
        if par_ == 0:
            L += ['\n分期五分組平均 Y\n', '| 期 | Q1 | Q2 | Q3 | Q4 | Q5 |', '|---|---|---|---|---|---|']
            for p in PER:
                m = (D.per == p).to_numpy()
                L.append(f'| {p} | ' + ' | '.join(pc(D[m & (q == g)].Y.mean()) for g in range(5)) + ' |')
        L.append('')
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
