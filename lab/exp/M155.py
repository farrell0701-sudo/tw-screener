# M-155：V4.1 事件中 R4 營收創高的確認（偶數代碼；只測一次）。定義與判準見 registry_M.md M-155（含執行前修訂）。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M154 import rev_month_factors, ic

H = 20; COST = 0.0077
PERIODS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2100-12-31'))
CUTS = (-0.25, -0.20, 0.0)          # 主、高原、並報
EDGES = [-0.2705, -0.1320, -0.0419, 0.0153]   # 探索集五分位切點（M-154，描述用）


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; daily = S['daily']
    U, signal = dbg['U'], dbg['signal']
    MD = daily['bear_mode'].to_numpy(bool); OFF = daily['off'].to_numpy(bool)
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    LA = np.asarray(F['LA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CEW = np.cumprod(1 + ew)
    sid = P['meta']['stock_id'].astype(str); par = (sid.str[-1].astype(int) % 2).to_numpy()
    skey = P['meta']['security_key'].to_numpy()
    R4, _ = rev_month_factors()

    rows = []
    for pname, a, b in PERIODS:
        t_lo = int(dates.searchsorted(pd.Timestamp(a))); t_hi = min(int(dates.searchsorted(pd.Timestamp(b), side='right')) - 1, T - 1)
        tt, jj = np.nonzero(signal)
        ok = (tt >= t_lo) & (tt + H <= t_hi); tt, jj = tt[ok], jj[ok]
        for t, j in zip(tt, jj):
            ent = OA[t + 1, j]
            if not (VALID[t + 1, j] and np.isfinite(ent) and ent > 0): continue
            y = Cf[t + H, j] / ent - 1 - (CEW[t + H] / CEW[t] - 1)
            seg = LA[t + 1:t + H + 1, j]; mae = np.nanmin(seg) / ent - 1 if np.isfinite(seg).any() else np.nan
            # 含停損版
            ex = None
            for d in range(t + 1, t + H + 1):
                if not VALID[d, j]: continue
                if OFF[d] or ((MD[d] or MD[t]) and CA[d, j] < ent * 0.95):
                    nx = d + 1
                    while nx < T and not VALID[nx, j]: nx += 1
                    ex = OA[nx, j] if nx < T and np.isfinite(OA[nx, j]) else CA[d, j]
                    break
            if ex is None: ex = Cf[t + H, j]
            per = dates[t].to_period('M') - (1 if dates[t].day > 10 else 2)
            try: r4 = float(R4.at[per, skey[j]])
            except KeyError: r4 = np.nan
            rows.append(dict(period=pname, par=int(par[j]), t=int(t), Y=float(y), MAE=float(mae), STOP=float(ex / ent - 1 - COST), R4=r4))
    X = pd.DataFrame(rows)

    def st(d):
        y = d.Y.to_numpy()
        if len(y) == 0: return dict(n=0)
        return dict(n=len(y), mean=y.mean(), med=np.median(y), win=(y > 0).mean(), succ=(y >= .05).mean(), fake=(y <= -.05).mean(),
                    lt=(y <= -.10).mean(), rt=(y >= .15).mean(), mae=np.nanmedian(d.MAE), stop=d.STOP.mean())

    def cut_test(d, c):
        rm = d.R4 < c; kp = ~rm                        # 缺值 → 保留
        a, k = st(d[rm]), st(d[kp])
        lt_share = (rm & (d.Y <= -.10)).sum() / max(1, (d.Y <= -.10).sum())
        per = {p: (d[rm & (d.period == p)].Y.mean(), d[kp & (d.period == p)].Y.mean()) for p, _, _ in PERIODS}
        ca = a['mean'] < k['mean'] and all(x < y for x, y in per.values())
        cb = a['rt'] <= k['rt']; cc = lt_share >= rm.mean(); cd = a['stop'] < k['stop']
        return dict(rm=a, kp=k, frac=rm.mean(), lt_share=lt_share, per=per, ok=dict(a=ca, b=cb, c=cc, d=cd), all=ca and cb and cc and cd)

    pc = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:+.2%}'
    pp = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:.0%}'
    yn = lambda b: '✓' if b else '✗'
    L = ['# M-155 結果：V4.1 事件 R4 營收創高確認集（偶數代碼）\n']
    verdict = {}
    for par_, lab in ((0, '偶數代碼（確認集，判準）'), (1, '奇數代碼（描述；2005–15 為探索集）')):
        D = X[X.par == par_]
        L.append(f'## {lab}\n')
        L.append(f"事件 {len(D)}（2005–15 {int((D.period == '2005–15').sum())}、2016–26 {int((D.period == '2016–26').sum())}）；R4 缺值 {D.R4.isna().mean():.0%}。\n")
        r, t, n = ic(D.R4.to_numpy(), D.Y.to_numpy())
        icp = {p: ic(D[D.period == p].R4.to_numpy(), D[D.period == p].Y.to_numpy()) for p, _, _ in PERIODS}
        L.append('| 範圍 | IC | t | n |'); L.append('|---|---|---|---|')
        L.append(f'| 2005–26 合併 | {r:+.3f} | {t:+.2f} | {n} |')
        for p, v in icp.items(): L.append(f'| {p} | {v[0]:+.3f} | {v[1]:+.2f} | {v[2]} |')
        c1 = t >= 2.0 and all(v[0] > 0 for v in icp.values())
        L.append(f"\n判準 1（合併 t ≥ 2.0 且兩期 IC > 0）：{yn(c1)}\n" if par_ == 0 else '')
        L += ['| 切點 | 剔除比例 | 被剔 n | 被剔平均 Y | 保留平均 Y | 05–15 被剔／保留 | 16–26 被剔／保留 | 被剔右尾 | 保留右尾 | 左尾占比（vs 剔除比例） | 被剔含停損 | 保留含停損 | (a)(b)(c)(d) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        res = {}
        for c in CUTS:
            z = cut_test(D, c); res[c] = z; o = z['ok']
            p1, p2 = z['per']['2005–15'], z['per']['2016–26']
            L.append(f"| {c:+.2f}{'（主）' if c == CUTS[0] else '（高原）' if c == CUTS[1] else '（並報）'} | {z['frac']:.0%} | {z['rm']['n']} | {pc(z['rm']['mean'])} | {pc(z['kp']['mean'])} "
                     f"| {pc(p1[0])}／{pc(p1[1])} | {pc(p2[0])}／{pc(p2[1])} | {pp(z['rm']['rt'])} | {pp(z['kp']['rt'])} | {pp(z['lt_share'])}（{pp(z['frac'])}） "
                     f"| {pc(z['rm']['stop'])} | {pc(z['kp']['stop'])} | {yn(o['a'])}{yn(o['b'])}{yn(o['c'])}{yn(o['d'])} |")
        if par_ == 0:
            c2 = res[CUTS[0]]['all'] and res[CUTS[1]]['all']
            L.append(f"\n判準 2（主 −0.25 與高原 −0.2 的 (a)–(d) 全過）：{yn(c2)}\n")
            L.append(f"**結論：{'確認（兩項判準皆過）' if c1 and c2 else '未確認，此方向結案'}**\n")
            verdict = dict(c1=c1, c2=c2)
        # 五分組（探索集切點）
        q = np.digitize(D.R4.fillna(np.inf), EDGES)       # 缺值放最後一欄（另列）
        L += ['五分組（探索集切點；缺值另列）\n', '| 組 | n | 平均 Y | 中位數 | 勝率 | 成功率 | 假突破率 | 左尾 | 右尾 | MAE 中位數 | 含停損 |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for g in range(5):
            m = (q == g) & D.R4.notna()
            s = st(D[m])
            L.append(f"| Q{g + 1} | {s['n']} | {pc(s['mean'])} | {pc(s['med'])} | {pp(s['win'])} | {pp(s['succ'])} | {pp(s['fake'])} | {pp(s['lt'])} | {pp(s['rt'])} | {pc(s['mae'])} | {pc(s['stop'])} |")
        s = st(D[D.R4.isna()])
        if s['n']: L.append(f"| 缺值 | {s['n']} | {pc(s['mean'])} | {pc(s['med'])} | {pp(s['win'])} | {pp(s['succ'])} | {pp(s['fake'])} | {pp(s['lt'])} | {pp(s['rt'])} | {pc(s['mae'])} | {pc(s['stop'])} |")
        L.append('')
    head = 'R4 確認集：' + ('確認' if verdict.get('c1') and verdict.get('c2') else f"未確認（判準1 {'過' if verdict.get('c1') else '不過'}、判準2 {'過' if verdict.get('c2') else '不過'}），結案")
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
