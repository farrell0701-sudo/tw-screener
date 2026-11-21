# M-157：V4.1 近 20 日訊號數多時跳過事件（2016–26 時期外驗證；只測一次）。定義與判準見 registry_M.md M-157。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41ev import events, removal, stat
sys.path.insert(0, str(Path(__file__).resolve().parent))
from M154 import ic

MAIN, PLATS = 27, (21, 33)
QE = [11.5, 17.5, 21.5, 27.0]       # 探索集五分位（≤11、12–17、18–21、22–26、≥27）


def main():
    X = events()
    D = X[X.year >= 2016].copy(); D['sub'] = np.where(D.year <= 2020, '2016–20', '2021–26')
    SUB = ['2016–20', '2021–26']
    pc = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:+.2%}'
    pp = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:.0%}'
    yn = lambda b: '✓' if b else '✗'

    def daily_ic(d):
        g = d.groupby('t').agg(x=('A22', 'first'), y=('Y', 'mean'))
        return ic(g.x.to_numpy(float), g.y.to_numpy(float))
    r, t, n = daily_ic(D); rs = {s: daily_ic(D[D['sub'] == s]) for s in SUB}
    c1 = t <= -2.0 and all(v[0] < 0 for v in rs.values())
    re_, te_, ne_ = ic(D.A22.to_numpy(float), D.Y.to_numpy())
    L = ['# M-157 結果：V4.1 近 20 日訊號數多時跳過事件（2016–26，奇偶代碼全部）\n',
         f"事件 {len(D)}（2016–20 {int((D['sub'] == '2016–20').sum())}、2021–26 {int((D['sub'] == '2021–26').sum())}）；事件日 {D.t.nunique()}。\n",
         '## 判準 1：每日平均 IC\n', '| 範圍 | IC | t | 日數 |', '|---|---|---|---|', f'| 2016–26 | {r:+.3f} | {t:+.2f} | {n} |']
    for s, v in rs.items(): L.append(f'| {s} | {v[0]:+.3f} | {v[1]:+.2f} | {v[2]} |')
    L.append(f'\n（事件層 IC 參考：{re_:+.3f}，t {te_:+.2f}）\n\n判準 1（t ≤ −2.0 且兩段 IC < 0）：{yn(c1)}\n')
    L += ['## 判準 2：跳過組 vs 保留組\n', '| 版本 | 跳過比例 | 跳過 n | 跳過平均 Y | 保留平均 Y | 2016–20 跳過／保留 | 2021–26 跳過／保留 | 跳過右尾 | 保留右尾 | 左尾占比（vs 跳過比例） | 跳過含停損 | 保留含停損 | (a)(b)(c)(d) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    R = {}
    for k in (MAIN,) + PLATS:
        z = removal(D, (D.A22 >= k).to_numpy(), 'sub', SUB); R[k] = z; o = z['ok']
        L.append(f"| ≥ {k}{'（主）' if k == MAIN else '（高原）'} | {z['frac']:.0%} | {z['rm']['n']} | {pc(z['rm']['mean'])} | {pc(z['kp']['mean'])} | {pc(z['per']['2016–20'][0])}／{pc(z['per']['2016–20'][1])} "
                 f"| {pc(z['per']['2021–26'][0])}／{pc(z['per']['2021–26'][1])} | {pp(z['rm']['rt'])} | {pp(z['kp']['rt'])} | {pp(z['lt_share'])}（{pp(z['frac'])}） | {pc(z['rm']['stop'])} | {pc(z['kp']['stop'])} | {yn(o['a'])}{yn(o['b'])}{yn(o['c'])}{yn(o['d'])} |")
    c2 = R[MAIN]['all'] and all(R[k]['ok']['a'] for k in PLATS)
    ok = c1 and c2
    L.append(f"\n判準 2（主版本 (a)–(d) 全過，高原兩版 (a) 皆過）：{yn(c2)}\n\n**結論：{'通過' if ok else '未通過，此方向結案'}**\n")
    q = np.digitize(D.A22, QE)
    L += ['## 2016–26 五分組（探索集切點）\n', '| 組 | 範圍 | n | 平均 Y | 中位數 | 勝率 | 假突破率 | 左尾 | 右尾 | MAE 中位數 | 含停損 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    lab = ['≤ 11', '12–17', '18–21', '22–26', '≥ 27']
    for g in range(5):
        s = stat(D[q == g])
        L.append(f"| Q{g + 1} | {lab[g]} | {s['n']} | {pc(s['mean'])} | {pc(s['med'])} | {pp(s['win'])} | {pp(s['fake'])} | {pp(s['lt'])} | {pp(s['rt'])} | {pc(s['mae'])} | {pc(s['stop'])} |")
    L += ['\n## 每年：被跳過比例與兩組平均 Y（主版本 ≥ 27）\n', '| 年 | 事件 | 跳過比例 | 跳過平均 Y | 保留平均 Y |', '|---|---|---|---|---|']
    for y, g in D.groupby('year'):
        m = g.A22 >= MAIN
        L.append(f'| {y} | {len(g)} | {m.mean():.0%} | {pc(g[m].Y.mean())} | {pc(g[~m].Y.mean())} |')
    E = X[(X.par == 0) & (X.year <= 2015)].copy(); E['sub'] = '2005–15'
    z = removal(E, (E.A22 >= MAIN).to_numpy(), 'sub', ['2005–15']); re2, te2, _ = daily_ic(E)
    L += ['\n## 並報：偶數代碼 2005–15（與探索集同日期，非獨立）\n',
          f"每日 IC {re2:+.3f}（t {te2:+.2f}）；≥ 27 跳過 {z['frac']:.0%}：跳過 {pc(z['rm']['mean'])}／保留 {pc(z['kp']['mean'])}；右尾 {pp(z['rm']['rt'])}／{pp(z['kp']['rt'])}；含停損 {pc(z['rm']['stop'])}／{pc(z['kp']['stop'])}。\n"]
    return dict(summary='\n'.join(L), headline='訊號數跳過 2016–26 驗證：' + ('通過' if ok else '未通過，結案'), n_variants=0)
