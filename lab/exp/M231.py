# M-231：L-131 機制拆解（交易層，100 路徑）。定義見 registry_M.md M-231。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M190

NPATH = 100
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: i for i, k in enumerate(keys)}
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S
    FE = M190.features(P, F, Tr); POS = FE['POS250']
    signals.EXTRA['POS'] = [POS.astype(np.float32)]
    signals.build('M231_P20', mode='fix', P=P, F=F, x_rev=('filter', 0.2, 'POS'))
    posp = pd.DataFrame(np.where(pool, POS, np.nan)).rank(axis=1, pct=True).to_numpy()
    ATRP = np.asarray(signals.load_sig('fix')['ATRP']); atrr = pd.DataFrame(np.where(U, ATRP, np.nan)).rank(axis=1, pct=True).to_numpy()
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill(); L = pd.DataFrame(np.asarray(F['LA'], np.float64)).ffill()
    r250 = (C / C.shift(250) - 1).to_numpy(); r20 = (C / C.shift(20) - 1).to_numpy()
    h250 = H.rolling(250, min_periods=200).max().to_numpy(); l250 = L.rolling(250, min_periods=200).min().to_numpy()
    # 過去 250 日內最大跌幅（滾動：收盤 ÷ 之前最高 − 1 的最小值）
    dd = (C / C.rolling(250, min_periods=200).max() - 1).rolling(250, min_periods=200).min().to_numpy()
    Cn = C.to_numpy()

    def trades(sig):
        R = sim.run_paths('fix', NPATH, sig=sig)
        rows = []
        for i in range(NPATH):
            for (code, d0, d1, w, ret, hold, why, st, lk) in R['X'][i][1]:
                d0 = pd.Timestamp(d0)
                if d0 < pd.Timestamp('2005-01-01'): continue
                j = kidx.get(str(code)); t = dates.get_loc(d0) - 1
                rows.append((i, j, d0, w, ret, hold, why, t))
        D = pd.DataFrame(rows, columns=['i', 'j', 'd0', 'w', 'ret', 'hold', 'why', 't'])
        tt, jj = D.t.to_numpy(), D.j.to_numpy()
        D['posp'] = posp[tt, jj]; D['atrr'] = atrr[tt, jj]; D['r250'] = r250[tt, jj]; D['r20'] = r20[tt, jj]; D['dd'] = dd[tt, jj]
        D['dh'] = Cn[tt, jj] / h250[tt, jj] - 1; D['dl'] = Cn[tt, jj] / l250[tt, jj] - 1; D['tld'] = Tr[tt, jj]
        return D
    B = trades('fix'); X = trades('M231_P20')
    B['excl'] = B.posp < 0.2
    keyB = set(zip(B.j, B.d0)); X['new'] = [(j, d) not in keyB for j, d in zip(X.j, X.d0)]
    yrs = lambda d: (d.d0.max() - d.d0.min()).days / 365.25
    def summ(nm, x, n_years):
        return (f"| {nm} | {len(x) / NPATH / n_years:.1f} | {x.ret.mean():+.2%} | {x.ret.median():+.2%} | {(x.ret > 0).mean():.0%} | {(x.why == '移動停損').mean():.0%} | "
                f"{x.hold.mean():.0f} | {(x.ret <= -0.10).mean():.0%} | {(x.ret >= 0.20).mean():.0%} |")
    Lg = ['# M-231 結果：L-131 機制拆解（X1 交易，100 路徑合計；2005 起）\n']
    for era, lo, hi in ERAS:
        b = B[(B.d0 >= lo) & (B.d0 <= hi)]; x = X[(X.d0 >= lo) & (X.d0 <= hi)]; ny = yrs(b)
        Lg += [f'## {era}\n', '### (A)(B) 交易結果\n', '| 組 | 每年筆數（每路徑） | 平均 | 中位 | 勝率 | 移動停損出場 | 持有日 | ≤ −10% | ≥ +20% |', '|---|---|---|---|---|---|---|---|---|',
               summ('新基準：全部', b, ny), summ('新基準：被剔除（POS 後 20%）', b[b.excl], ny), summ('新基準：保留', b[~b.excl], ny),
               summ('P20：全部', x, ny), summ('P20：換進來的（新基準沒有）', x[x.new], ny), '']
        e, k = b[b.excl], b[~b.excl]
        Lg += ['### (C) 被剔除 vs 保留的進場特徵（中位數）\n', '| 組 | 進場前 250 日報酬 | 前段下跌比例（250 日報酬 < 0） | 過去 250 日最大跌幅 | 距 250 日高 | 距 250 日低 | ATR% 排名 | 「太抖」比例 | T_ld | 近 20 日漲幅 |', '|---|---|---|---|---|---|---|---|---|---|']
        for nm, y in (('被剔除', e), ('保留', k)):
            Lg.append(f"| {nm} | {y.r250.median():+.1%} | {(y.r250 < 0).mean():.0%} | {y.dd.median():.1%} | {y.dh.median():+.1%} | {y.dl.median():+.1%} | {y.atrr.median():.2f} | {(y.atrr >= 0.485).mean():.0%} | {y.tld.median():.2f} | {y.r20.median():+.1%} |")
        Lg += ['', '### (D) 假說檢驗：被剔除交易依「前段下跌」與「太抖」分組\n', '| 組 | 筆數占比 | 平均 | 中位 | 勝率 | 移動停損 | ≥ +20% |', '|---|---|---|---|---|---|---|']
        for nm, y in (('被剔除｜前段下跌（250 日報酬 < 0）', e[e.r250 < 0]), ('被剔除｜前段上漲', e[e.r250 >= 0]), ('被剔除｜太抖', e[e.atrr >= 0.485]), ('被剔除｜不抖', e[e.atrr < 0.485]),
                      ('保留｜前段下跌', k[k.r250 < 0]), ('保留｜太抖', k[k.atrr >= 0.485])):
            Lg.append(f"| {nm} | {len(y) / max(len(e if nm.startswith('被') else k), 1):.0%} | {y.ret.mean():+.2%} | {y.ret.median():+.2%} | {(y.ret > 0).mean():.0%} | {(y.why == '移動停損').mean():.0%} | {(y.ret >= 0.20).mean():.0%} |")
        Lg.append('')
    # (E) 逐年
    B['y'] = B.d0.dt.year; X['y'] = X.d0.dt.year
    ye = B[B.excl].groupby('y').apply(lambda g: (g.w * g.ret).sum() / NPATH); yn = X[X.new].groupby('y').apply(lambda g: (g.w * g.ret).sum() / NPATH)
    yy = pd.concat([ye.rename('ex'), yn.rename('nw')], axis=1).fillna(0); yy['net'] = yy.nw - yy.ex
    Lg += ['## (E) 逐年：被剔除交易損益（新基準）vs 換進交易損益（P20）（每路徑、以 X1 權益比例計）\n', '| 年 | ' + ' | '.join(str(y) for y in yy.index) + ' |', '|' + '---|' * (len(yy) + 1),
           '| 被剔除 | ' + ' | '.join(f'{v:+.1%}' for v in yy.ex) + ' |', '| 換進 | ' + ' | '.join(f'{v:+.1%}' for v in yy.nw) + ' |', '| 淨差 | ' + ' | '.join(f'{v:+.1%}' for v in yy.net) + ' |']
    return dict(summary='\n'.join(Lg), headline='L-131 機制拆解', n_variants=0)
