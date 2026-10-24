"""第二批：趨勢類因子深入
問題 1 表達方式：同一個「EMA12 與 EMA200 距離」，除以 ATR、除以股價、除以波動率，哪個對？
問題 2 週期：快慢線長度是不是一片高原（不是剛好 12/200 最好）
問題 3 趨勢的其他面向：品質（平滑度）、成熟度（多久了）、延伸（離短均線多遠）、加速度、產業相對
方向都在看數據前宣告。"""
import numpy as np, pandas as pd
from factors import base, ind_relative


def ema(C, s): return C.ewm(span=s, adjust=False).mean()


def bars_since(cond):
    """條件連續成立的天數（不成立 = 0）"""
    c = cond.astype(float).fillna(0)
    cs = c.cumsum()
    return cs - cs.where(c == 0).ffill().fillna(0)


def rolling_r2_slope(logC, n=120):
    """對 log 價格做 n 日線性回歸：回傳 R² × sign(斜率)（越接近 +1 = 越平滑的上升）"""
    t = np.arange(n, dtype=float); t = t - t.mean(); stt = (t ** 2).sum()
    y = logC
    # 用 rolling 均值與 rolling 協方差的封閉解
    k = pd.DataFrame(np.arange(len(y), dtype=float)[:, None].repeat(y.shape[1], 1), index=y.index, columns=y.columns)
    S1 = y.rolling(n).sum(); K = (y * k).rolling(n).sum()
    sty = K - (k - n + 1) * S1 - (n - 1) / 2 * S1      # Σ (i − 平均 i)·y
    b = sty / stt
    syy = y.rolling(n).var(ddof=0) * n
    r2 = (b ** 2 * stt) / syy
    return r2.clip(0, 1) * np.sign(b)


def build2(P, grid=True):
    C, H, L = P['C'], P['H'], P['L']
    B = base(P); atr, sig60 = B['atr'], B['sig60']
    e = {s: ema(C, s) for s in (5, 12, 20, 40, 60, 100, 150, 200, 250)}
    F = {}
    def add(name, X, sign, why): F[name] = (X.astype('float32'), sign, why)
    ld = (e[12] - e[200]) / atr
    add('T_ld', ld, +1, '基準：(EMA12−EMA200)/ATR')
    # 問題 1：表達方式
    add('E_pct', (e[12] - e[200]) / C, +1, '同距離 ÷ 股價（百分比）')
    add('E_sig', (e[12] - e[200]) / (C * sig60), +1, '同距離 ÷ (股價×60日日報酬標準差)')
    add('E_raw_ratio', np.log(e[12] / e[200]), +1, 'log(EMA12/EMA200)，等同百分比')
    # 問題 3：其他面向
    lc = np.log(C)
    add('Q_r2', rolling_r2_slope(lc, 120), +1, '120 日 log 價格回歸 R²×方向：趨勢平滑度')
    net = (C - C.shift(60)).abs(); path = C.diff().abs().rolling(60).sum()
    add('Q_er', np.sign(C - C.shift(60)) * net / path, +1, '60 日效率比×方向：淨移動 ÷ 總路徑')
    add('Q_above20', (C > e[20]).astype(float).rolling(60).mean(), +1, '近 60 日收在 EMA20 之上的比例')
    add('A_age200', np.log1p(bars_since(C > e[200])), -1, '連續站上 EMA200 的天數（log）：越新的趨勢越好？')
    align = (e[12] > e[20]) & (e[20] > e[60]) & (e[60] > e[200])
    add('A_alignage', np.log1p(bars_since(align)), -1, '均線多頭排列持續天數（log）')
    add('X_ext20', (C - e[20]) / atr, -1, '收盤離 EMA20 幾個 ATR：短線過度延伸')
    add('V_acc', ld - ld.shift(20), +1, 'T_ld 的 20 日變化：趨勢在加速')
    add('I_ld', ind_relative(ld, P), +1, '產業相對 T_ld')
    if grid:
        for f in (5, 12, 20, 40):
            for s in (100, 150, 200, 250):
                add(f'G_{f}_{s}', (e[f] - e[s]) / atr, +1, f'(EMA{f}−EMA{s})/ATR')
    return F
