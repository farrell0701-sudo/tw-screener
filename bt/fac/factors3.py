"""第三批：相對強度類
在橫斷面排序裡，「相對大盤」的報酬和原始報酬排名相同（大盤報酬每天對每檔都一樣），
所以相對強度要有意義，必須換個角度：扣掉 beta 的殘差、逆勢時的表現、RS 線位置、族群強度、持續性。
方向都在看數據前宣告。"""
import numpy as np, pandas as pd
from factors import base


def tld_floor(P, q=0.05):
    """T_ld 正式版：ATR 下限 = 當日全市場 ATR% 的 q 分位 × 股價（排除收購案等價格被釘住的股票）"""
    C = P['C']; atr = base(P)['atr']
    e12 = C.ewm(span=12, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
    fl = (atr / C).quantile(q, axis=1)
    return (e12 - e200) / np.maximum(atr, C.mul(fl, axis=0))


def build3(P):
    C = P['C']; B = base(P); sig60, sig250 = B['sig60'], B['sig250']
    r = np.log(C / C.shift(1))
    m = np.log(P['taiex'] / P['taiex'].shift(1)).reindex(C.index)
    F = {}
    def add(name, X, sign, why): F[name] = (X.astype('float32'), sign, why)
    add('T_ldf', tld_floor(P), +1, '基準：T_ld（含 ATR 下限）')
    # 1. 殘差動能：扣掉大盤 beta 後的個股自身表現
    mv = m.rolling(250, min_periods=120).var()
    beta = r.rolling(250, min_periods=120).cov(m) .div(mv, axis=0) if False else \
        (r.mul(m, axis=0).rolling(250, min_periods=120).mean() - r.rolling(250, min_periods=120).mean().mul(m.rolling(250, min_periods=120).mean(), axis=0)).div(mv, axis=0)
    e = r - beta.shift(1).mul(m, axis=0)                       # 用昨天以前估的 beta
    es = e.shift(20).rolling(100, min_periods=70)
    add('R_resid', es.sum() / (es.std() * np.sqrt(100)), +1, '殘差動能：扣 beta 後 t-120～t-20 的累積殘差 ÷ 殘差波動')
    e60 = e.rolling(60, min_periods=40)
    add('R_resid60', e60.sum() / (e60.std() * np.sqrt(60)), +1, '近 60 日殘差動能（不跳過最近一個月）')
    add('R_beta', beta, -1, 'beta：高 beta 股長期報酬較差（低 beta 異常）')
    # 2. 逆勢強度：大盤下跌日的超額報酬
    down = m < 0
    ex = r.sub(m, axis=0)
    dn = ex.where(down.values[:, None].repeat(C.shape[1], 1))
    add('R_downday', dn.rolling(120, min_periods=30).mean() / sig60, +1, '近 120 日大盤下跌日的平均超額報酬（波動標準化）：逆勢抗跌')
    up = ex.where((~down).values[:, None].repeat(C.shape[1], 1))
    add('R_upday', up.rolling(120, min_periods=30).mean() / sig60, +1, '大盤上漲日的平均超額報酬（對照）')
    # 3. RS 線（個股/大盤）離 250 日高點多遠
    rs = np.log(C.div(P['taiex'], axis=0))
    add('R_rshigh', (rs - rs.rolling(250, min_periods=150).max()) / sig250, +1, 'RS 線距 250 日高點（波動單位）：RS 創新高')
    # 4. 持續性：近 120 日中，5 日報酬贏大盤的比例
    ex5 = (np.log(C / C.shift(5))).sub(np.log(P['taiex'] / P['taiex'].shift(5)), axis=0)
    add('R_consist', (ex5 > 0).astype(float).where(ex5.notna()).rolling(120, min_periods=60).mean(), +1, '近 120 日 5 日報酬贏大盤的比例：穩定跑贏')
    # 5. 族群強度
    ind = P['industry'].reindex(C.columns).fillna('NA').values
    r60va = np.log(C / C.shift(60)) / (sig60 * np.sqrt(60))
    add('R_ind60', r60va.T.groupby(ind).transform('median').T, +1, '所屬產業 60 日風險調整報酬中位數：族群動能')
    add('R_indld', tld_floor(P).T.groupby(ind).transform('median').T, +1, '所屬產業 T_ld 中位數：族群趨勢')
    # 6. 短期反轉
    add('R_rev5', np.log(C / C.shift(5)) / (sig60 * np.sqrt(5)), -1, '近 5 日報酬（波動標準化）：短線反轉')
    # 7. 52 週高點接近度（百分比版，George-Hwang）
    add('R_52wh', C / P['H'].rolling(250, min_periods=150).max(), +1, '收盤 ÷ 250 日最高價')
    return F
