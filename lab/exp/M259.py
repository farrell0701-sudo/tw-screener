# M-259：書中大盤轉折規則套用到 T_ld 強勢池（剔除超低波動）。定義見 registry_M.md M-259。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M258
from L155 import bull_line


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    pool = np.asarray(dbg['pool'])
    if pool.shape[1] != N:
        o = np.zeros((T, N), bool); o[:, cols] = pool; pool = o
    del S
    VALID = np.asarray(F['VALID']); CA = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); ATR = np.asarray(F['ATR'], np.float64)
    with np.errstate(all='ignore'):
        R = CA / np.vstack([np.full((1, N), np.nan), CA[:-1]]) - 1; ap = ATR / CA
    R[~VALID] = np.nan; R[np.abs(R) > 0.11] = np.nan     # 台股漲跌幅 10%，超出視為資料問題
    apr = pd.DataFrame(np.where(pool, ap, np.nan)).rank(axis=1, pct=True).to_numpy()
    mem = pool & ~(apr <= 0.10)
    memp = np.vstack([np.zeros((1, N), bool), mem[:-1]])
    with np.errstate(all='ignore'): pr = np.nanmean(np.where(memp & np.isfinite(R), R, np.nan), axis=1)
    pr = np.nan_to_num(pr)
    s = P['index']['TAIEX_P'].reindex(dates).ffill(); ok = s.notna().to_numpy()
    d_ = dates[ok]; C = s.to_numpy()[ok]; r = pr[ok]
    H, L = M258.index_hlc(d_); H = np.maximum(H.to_numpy(), C); L = np.minimum(L.to_numpy(), C)
    p0 = 600 / 7738
    V = {'無閘門（強勢池持有）': np.zeros(len(C), bool), 'V0 目前定義': bull_line(C, d_),
         'B1 只有正常轉折': M258.book_states(d_, C, H, L, monthend=False), 'B2 正常＋月底原則': M258.book_states(d_, C, H, L),
         '主 A 書字面 600 點': M258.book_states(d_, C, H, L, thr=600), '主 B 比例 7.75%': M258.book_states(d_, C, H, L, pct=p0),
         '主 B｜5%': M258.book_states(d_, C, H, L, pct=0.05), '主 B｜10%': M258.book_states(d_, C, H, L, pct=0.10),
         '主 B｜收盤價': M258.book_states(d_, C, C, C, pct=p0)}
    Lg = ['# M-259 結果：大盤轉折規則套用到強勢池（剔除最低 10% 波動，等權，空頭時曝險 20%）\n',
          f'股池平均每日檔數 {memp[ok].sum(1)[memp[ok].sum(1) > 0].mean():.0f}；不計股池換股成本。\n']
    res = {}
    for era, lo, hi in M258.ERA:
        Lg += [f'## {era}\n', '| 版本 | 年化 | 最大回撤 | Calmar | 每年翻轉 | 空頭天數 |', '|---|---|---|---|---|---|']
        for nm, st in V.items():
            x = M258.evalu(st, r, C, d_, lo, hi); res[(era, nm)] = x
            Lg.append(f"| {nm} | {x['cagr']:+.1%} | {x['mdd']:.1%} | {x['cal']:.2f} | {x['fl']:.1f} | {x['bear']:.0%} |")
        Lg.append('')
    yrs = [2008, 2011, 2015, 2018, 2020, 2022, 2024, 2025, 2026]
    Lg += ['## 關鍵年份報酬\n', '| 版本 | ' + ' | '.join(map(str, yrs)) + ' |', '|' + '---|' * (len(yrs) + 1)]
    for nm, st in list(V.items())[:6]:
        expo = np.where(np.r_[False, st[:-1]], 0.2, 1.0); ret = pd.Series(expo * r - np.r_[0, np.abs(np.diff(expo))] * M258.COST, index=d_)
        y = (1 + ret).groupby(d_.year).prod() - 1
        Lg.append(f'| {nm} | ' + ' | '.join(f'{y.get(k, np.nan):+.0%}' for k in yrs) + ' |')
    v0 = 'V0 目前定義'; out = []
    for mn in ('主 A 書字面 600 點', '主 B 比例 7.75%', 'B1 只有正常轉折'):
        w = [res[(e, mn)]['cal'] > res[(e, v0)]['cal'] for e, _, _ in M258.ERA]
        c = '三期都優於 V0' if all(w) else ('只有出版後較好' if w[2] and not all(w) else f'{sum(w)}/3 期優於 V0')
        out.append(f"{mn}：{c}（Calmar " + '／'.join(f"{res[(e, mn)]['cal']:.2f} vs {res[(e, v0)]['cal']:.2f}" for e, _, _ in M258.ERA) + '）')
    Lg += ['', '判準：' + '；'.join(out) + '（B1 為觀察）']
    h = 'M-259 強勢池：' + '；'.join(o.split('（Calmar')[0] for o in out)
    return dict(summary='\n'.join(Lg[:1] + [f'**{h}**\n'] + Lg[1:]), headline=h, n_variants=0)
