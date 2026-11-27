# M-258：書中大盤轉折規則（正常轉折＋月底原則＋低轉折）vs 目前多空點。定義見 registry_M.md M-258。
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel, twdata_path
from L155 import bull_line

COST = 0.00385
ERA = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–23.08（出版前）', '2016-01-01', '2023-08-31'), ('2023.09–26.11（出版後）', '2023-09-01', '2026-12-31'))


def index_hlc(dates):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(twdata_path() / 'staged' / 'TaiwanStockPrice_index' / '*.parquet')))])
    d = d[d.stock_id == 'TAIEX'].copy(); d['date'] = pd.to_datetime(d['date']); d = d.drop_duplicates('date', keep='last').set_index('date').sort_index()
    for c in ('high', 'low', 'close', 'max', 'min'): d[c] = pd.to_numeric(d[c], errors='coerce')
    H = d['max'].where(d['max'] > 0, d['high']); L = d['min'].where(d['min'] > 0, d['low'])
    return H.reindex(dates).ffill(), L.reindex(dates).ffill()


def book_states(d_, C, H, L, thr=None, pct=None, early=True, monthend=True, highturn=False):
    T = len(C); m = d_.to_period('M'); day = d_.day.to_numpy()
    mh = pd.Series(H, index=d_).groupby(m).max(); ml = pd.Series(L, index=d_).groupby(m).min(); per = list(mh.index); pi = {p: i for i, p in enumerate(per)}
    def th(x): return thr if thr is not None else (pct * x if pct is not None else None)
    bear = np.zeros(T, bool); line_out = np.full(T, np.nan)
    hm = lm = None; cur = None
    for t in range(T):
        k = pi[m[t]]
        if m[t] != cur: cur = m[t]; hm, lm = H[t], L[t]
        else: hm, lm = max(hm, H[t]), min(lm, L[t])
        if k < 2: continue
        H1, L1, H2, L2 = mh.iloc[k - 1], ml.iloc[k - 1], mh.iloc[k - 2], ml.iloc[k - 2]
        Nm = (max(H1, H2) + min(L1, L2)) / 2
        lm_ = Nm
        if th(Nm) is not None:                                     # 上個月的低（高）轉折影響本月
            if L1 - Nm < -th(Nm): lm_ = (H1 + L1) / 2
            elif highturn and H1 - Nm > th(Nm): lm_ = (H1 + L1) / 2
        NX = (max(H1, hm) + min(L1, lm)) / 2; nx_ = NX; cond = False
        if th(NX) is not None:
            if lm - NX < -th(NX): nx_ = (hm + lm) / 2; cond = True
            elif highturn and hm - NX > th(NX): nx_ = (hm + lm) / 2
        start = 5 if (early and cond) else 19
        ln = nx_ if (monthend and day[t] >= start) else lm_
        line_out[t] = ln; bear[t] = C[t] < ln
    return bear


def evalu(st, r, px, d_, lo, hi):
    T = len(px); i0, i1 = d_.searchsorted(pd.Timestamp(lo)), d_.searchsorted(pd.Timestamp(hi))
    expo = np.where(np.r_[False, st[:-1]], 0.2, 1.0); chg = np.r_[0, np.abs(np.diff(expo))] * COST
    ret = expo * r - chg; R = pd.Series(ret[i0:i1]); eq = (1 + R).cumprod()
    ny = len(R) / 245; cagr = eq.iloc[-1] ** (1 / ny) - 1; mdd = (eq / eq.cummax() - 1).min()
    fl = (np.diff(st[i0:i1].astype(int)) != 0).sum() / ny
    segs = []; t = max(i0, 1)
    while t < i1:
        if st[t] and not st[t - 1]:
            e = t
            while e + 1 < i1 and st[e + 1]: e += 1
            end = min(e + 1, T - 1); segs.append(-0.8 * (px[end] / px[t] - 1) - 0.8 * 2 * COST); t = end + 1
        else: t += 1
    sg = np.array(segs) if segs else np.zeros(0)
    return dict(cagr=cagr, mdd=mdd, cal=cagr / abs(mdd), fl=fl, bear=st[i0:i1].mean(), nr=(sg > 0).sum(), nw=(sg <= 0).sum(), br=sg[sg > 0].sum(), bw=sg[sg <= 0].sum())


def main():
    P = load_panel(); dates = P['dates']
    s = P['index']['TAIEX_P'].reindex(dates).ffill().dropna(); d_ = s.index; C = s.to_numpy(); T = len(C)
    Hs, Ls = index_hlc(d_); H = Hs.to_numpy(); L = Ls.to_numpy()
    H = np.where(np.isfinite(H), np.maximum(H, C), C); L = np.where(np.isfinite(L), np.minimum(L, C), C)
    r = np.r_[0, C[1:] / C[:-1] - 1]
    p0 = 600 / 7738
    V = {'V0 目前定義（收盤、月底原則從 1 日）': bull_line(C, d_),
         'B1 只有正常轉折': book_states(d_, C, H, L, monthend=False),
         'B2 正常＋月底原則 19 日': book_states(d_, C, H, L),
         '主 A 書字面（＋低轉折 600 點）': book_states(d_, C, H, L, thr=600),
         '主 B 比例版（＋低轉折 7.75%）': book_states(d_, C, H, L, pct=p0),
         '主 B｜門檻 5%': book_states(d_, C, H, L, pct=0.05),
         '主 B｜門檻 10%': book_states(d_, C, H, L, pct=0.10),
         '主 B｜不提前（只 19 日）': book_states(d_, C, H, L, pct=p0, early=False),
         '主 B｜加對稱高轉折': book_states(d_, C, H, L, pct=p0, highturn=True),
         '主 B｜收盤價高低點': book_states(d_, C, C, C, pct=p0)}
    Lg = ['# M-258 結果：書中大盤轉折規則 vs 目前多空點（指數層，空頭時曝險 20%）\n']
    res = {}
    for era, lo, hi in ERA:
        Lg += [f'## {era}\n', '| 版本 | 年化 | 最大回撤 | Calmar | 每年翻轉 | 空頭天數 | 真跌段 | 誤判段 | 真跌效益 | 誤判效益 |', '|---|---|---|---|---|---|---|---|---|---|']
        i0, i1 = d_.searchsorted(pd.Timestamp(lo)), d_.searchsorted(pd.Timestamp(hi)); eq = (1 + pd.Series(r[i0:i1])).cumprod(); ny = (i1 - i0) / 245
        cg = eq.iloc[-1] ** (1 / ny) - 1; md = (eq / eq.cummax() - 1).min()
        Lg.append(f'| 指數持有 | {cg:+.1%} | {md:.1%} | {cg / abs(md):.2f} | — | — | — | — | — | — |')
        for nm, st in V.items():
            x = evalu(st, r, C, d_, lo, hi); res[(era, nm)] = x
            Lg.append(f"| {nm} | {x['cagr']:+.1%} | {x['mdd']:.1%} | {x['cal']:.2f} | {x['fl']:.1f} | {x['bear']:.0%} | {x['nr']} | {x['nw']} | {x['br']:+.1%} | {x['bw']:+.1%} |")
        Lg.append('')
    # 年度（關鍵年份）
    yrs = [2008, 2011, 2015, 2018, 2020, 2022, 2024, 2025, 2026]
    Lg += ['## 關鍵年份報酬（指數層）\n', '| 版本 | ' + ' | '.join(map(str, yrs)) + ' |', '|' + '---|' * (len(yrs) + 1)]
    for nm, st in list(V.items())[:5] + [('指數持有', np.zeros(T, bool))]:
        expo = np.where(np.r_[False, st[:-1]], 0.2, 1.0); ret = pd.Series(expo * r - np.r_[0, np.abs(np.diff(expo))] * COST, index=d_)
        y = (1 + ret).groupby(d_.year).prod() - 1
        Lg.append(f'| {nm} | ' + ' | '.join(f'{y.get(k, np.nan):+.0%}' for k in yrs) + ' |')
    # 目前狀態
    last = d_[-1]
    Lg += ['', f"資料最後一日 {last:%Y-%m-%d} 收盤 {C[-1]:,.0f}：" + '；'.join(f"{nm.split(' ')[0]} {'空頭' if st[-1] else '多頭'}" for nm, st in list(V.items())[:5])]
    v0 = list(V)[0]; out = []
    for mn in ('主 A 書字面（＋低轉折 600 點）', '主 B 比例版（＋低轉折 7.75%）'):
        w = [res[(e, mn)]['cal'] > res[(e, v0)]['cal'] for e, _, _ in ERA]
        c = '優於目前定義' if all(w) else ('只有出版後較好（觀察）' if w[2] else '不採用')
        out.append(f"{mn.split('（')[0]}：{c}（Calmar " + '／'.join(f"{res[(e, mn)]['cal']:.2f} vs {res[(e, v0)]['cal']:.2f}" for e, _, _ in ERA) + '）')
    Lg += ['', '判準：' + '；'.join(out)]
    h = 'M-258 ' + '；'.join(o.split('（Calmar')[0] for o in out)
    return dict(summary='\n'.join(Lg[:1] + [f'**{h}**\n'] + Lg[1:]), headline=h, n_variants=0)
