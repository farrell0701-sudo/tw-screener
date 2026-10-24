"""因子評估工具
- 股票池：當日有價、上市滿 250 根、20 日均成交額 ≥ 當日全市場 20 分位（橫斷面，不用固定金額）
- 前瞻報酬：t+1 收盤進場（當天必須有成交），持有 h 日；停牌/下市以最後價計（不丟掉下市股）
- 波動調整：log 報酬 ÷ (過去 60 日日報酬標準差 × √h)
- 每個因子必須事先宣告預期方向 sign（+1 越大越好，-1 越小越好），報表一律乘上 sign
"""
import numpy as np, pandas as pd

DEV_START = '2016-01-01'
HORIZONS = (20, 60)
MIN_N = 100


def prepare(P):
    C, AMT = P['C'], P['AMT']
    age = C.notna().cumsum()
    amt20 = AMT.rolling(20, min_periods=15).mean()
    liq = amt20.ge(amt20.quantile(0.20, axis=1), axis=0)
    univ = C.notna() & (age >= 250) & liq
    lr = np.log(C / C.shift(1))
    sig60 = lr.rolling(60, min_periods=40).std()
    Cf = C.ffill()
    ent = C.shift(-1)                                   # t+1 收盤，必須是真實成交
    fwd = {}
    for h in HORIZONS:
        ex = Cf.shift(-(1 + h))
        r = np.log(ex / ent)
        r = r.where(ex.notna() & ent.notna())
        # 最後 h+1 天沒有完整的未來資料
        r.iloc[-(h + 1):] = np.nan
        # 先減去當日股票池平均（去掉大盤漂移），再除以個股波動；
        # 否則多頭年代低波動股會被「放大的正漂移」機械性地加分
        ex_r = r.sub(r.where(univ).mean(1), axis=0)
        fwd[h] = dict(raw=r, va=ex_r / (sig60 * np.sqrt(h)))
    dates = C.index[C.index >= DEV_START]
    return dict(univ=univ, fwd=fwd, dates=dates, sig60=sig60)


def _row_spearman(X, Y):
    m = X.notna() & Y.notna()
    X, Y = X.where(m), Y.where(m)
    rx, ry = X.rank(axis=1), Y.rank(axis=1)
    dx = rx.sub(rx.mean(1), axis=0); dy = ry.sub(ry.mean(1), axis=0)
    ic = (dx * dy).sum(1) / np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
    return ic.where(m.sum(1) >= MIN_N)


def _nw_t(x, lag):
    x = x.dropna().values; n = len(x)
    if n < lag * 3: return np.nan
    e = x - x.mean(); v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def neutralize(F, ctrls):
    """每日把因子排名對控制變數排名做回歸，取殘差（排除低波動、規模等已知效應）"""
    R = F.rank(axis=1, pct=True).values
    Cs = [c.reindex_like(F).where(F.notna()).rank(axis=1, pct=True).values for c in ctrls]
    out = np.full(R.shape, np.nan)
    for i in range(R.shape[0]):
        y = R[i]; X = np.column_stack([np.ones_like(y)] + [c[i] for c in Cs])
        m = ~np.isnan(y) & ~np.isnan(X).any(1)
        if m.sum() < MIN_N: continue
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        out[i, m] = y[m] - X[m] @ b
    return pd.DataFrame(out, index=F.index, columns=F.columns)


def evaluate(name, F, sign, E, P, ctrls=None):
    """回傳 (摘要 dict, 逐年 IC DataFrame)"""
    d = E['dates']
    F = (F.reindex(index=P['C'].index, columns=P['C'].columns) * sign).where(E['univ']).loc[d]
    out = dict(factor=name, sign=sign, n=float(F.notna().sum(1).median()))
    yearly = {}
    for h in HORIZONS:
        raw, va = E['fwd'][h]['raw'].loc[d], E['fwd'][h]['va'].loc[d]
        ic_r, ic_v = _row_spearman(F, raw), _row_spearman(F, va)
        out[f'IC{h}'] = ic_r.mean(); out[f'ICva{h}'] = ic_v.mean()
        out[f'IR{h}'] = ic_v.mean() / ic_v.std()
        out[f't{h}'] = _nw_t(ic_v, h)
        y = ic_v.groupby(ic_v.index.year).mean(); yearly[h] = y
        out[f'正年{h}'] = f'{int((y > 0).sum())}/{int(y.notna().sum())}'
        if ctrls is not None:
            if h == HORIZONS[0]: Fn = neutralize(F, [c.loc[d] for c in ctrls])
            icn = _row_spearman(Fn, va); out[f'ICn{h}'] = icn.mean(); out[f'tn{h}'] = _nw_t(icn, h)
        out[f'前半{h}'] = ic_v[:'2020'].mean(); out[f'後半{h}'] = ic_v['2021':].mean()
        # 五分位：相對當日股票池平均的超額 log 報酬（%／h 日）
        m = F.notna() & raw.notna()
        q = F.where(m).rank(axis=1, pct=True)
        ex = raw.where(m).sub(raw.where(m).mean(1), axis=0)
        qm = [ex.where((q > i / 5) & (q <= (i + 1) / 5)).mean(1) for i in range(5)]
        for i in range(5): out[f'Q{i + 1}_{h}'] = qm[i].mean() * 100
        out[f'Q5-Q1_{h}'] = (qm[4] - qm[0]).mean() * 100
        if h == 20:
            mo = qm[4].groupby(qm[4].index.to_period('M')).mean().dropna()
            pos = mo.clip(lower=0).sort_values(ascending=False)
            out['Q5正月%'] = (mo > 0).mean() * 100
            # 最好的 10% 月份貢獻了多少比例的累計超額（越高 = 越集中在少數時段）
            out['Q5集中度'] = pos.iloc[:max(1, len(mo) // 10)].sum() / mo.sum() if mo.sum() > 0 else np.nan
            out['Q5最差年'] = (qm[4].groupby(qm[4].index.year).mean() * 100).min()
    s = F.iloc[::20]
    out['自相關20'] = np.nanmean([_row_spearman(s.iloc[[i]], s.iloc[[i + 1]].set_axis(s.index[[i]])).iloc[0]
                               for i in range(len(s) - 1)])
    Y = pd.DataFrame({f'{name}|{h}': yearly[h] for h in HORIZONS})
    return out, Y
