"""第一批因子：全部寫成無尺度形式（ATR 單位、比值、波動標準化）
每個因子 = (函數, 預期方向, 邏輯)。方向在看數據前就宣告。"""
import numpy as np, pandas as pd


def base(P):
    C, H, L = P['C'], P['H'], P['L']
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e = {s: C.ewm(span=s, adjust=False).mean() for s in (12, 20, 60, 120, 200)}
    lr = np.log(C / pc)
    sig60 = lr.rolling(60, min_periods=40).std()
    sig250 = lr.rolling(250, min_periods=150).std()
    return dict(atr=atr, e=e, lr=lr, sig60=sig60, sig250=sig250)


def ind_relative(X, P):
    """減去同產業當日中位數（產業相對）"""
    ind = P['industry'].reindex(X.columns).fillna('NA')
    return X - X.T.groupby(ind.values).transform('median').T


def build(P, daily=None):
    C, H, L, AMT = P['C'], P['H'], P['L'], P['AMT']
    B = base(P); atr, e = B['atr'], B['e']
    r = lambda n: np.log(C / C.shift(n))
    F = {}
    def add(name, X, sign, why): F[name] = (X.astype('float32'), sign, why)

    # ── 趨勢 ──
    add('T_ld', (e[12] - e[200]) / atr, +1, 'EMA12 與 EMA200 距離（ATR 單位）：多頭排列的強度')
    add('T_px200', (C - e[200]) / atr, +1, '收盤在長期均線上方多少個 ATR')
    add('T_slope200', (e[200] - e[200].shift(20)) / atr, +1, '長期均線 20 日斜率（ATR 單位）：長期趨勢方向')
    add('T_align', sum((e[a] > e[b]).astype(float) for a, b in ((12, 20), (20, 60), (60, 120), (120, 200))),
        +1, '均線多頭排列的層數（0–4）')
    add('T_dhi250', (C - H.rolling(250).max()) / atr, +1, '距 250 日高點幾個 ATR：越接近新高越強')
    # ── 動能／相對強度 ──
    add('M_r20', r(20), -1, '20 日報酬：短期反轉（台股短線過熱常回吐）')
    add('M_r60', r(60), +1, '60 日報酬：中期動能')
    add('M_r120', r(120), +1, '120 日報酬')
    add('M_12_1', np.log(C.shift(20) / C.shift(250)), +1, '經典 12-1 月動能（跳過最近一個月）')
    add('M_r60va', r(60) / (B['sig60'] * np.sqrt(60)), +1, '風險調整後 60 日動能：同樣漲幅，波動越小越可信')
    add('M_r60ind', ind_relative(r(60), P), +1, '產業相對 60 日動能：排除整個族群一起動')
    # ── 結構 ──
    if daily is not None:
        add('S_ovh', daily['ovh'], -1, '上方套牢量占比：越少越容易上漲')
        add('S_loratio', daily['lo_ratio'], -1, '近一年低點 ÷ 前一年低點：<1 代表破前低後打底（洗盤）')
        add('S_drop20va', daily['drop20'] / (B['sig250'] * np.sqrt(20)), -1, '近 300 根最急 20 日跌幅（波動標準化）：急殺後的反彈潛力')
        add('S_near5', daily['near5'], +1, '近 20 根重疊度：近期收斂')
        add('S_far5', daily['far5'], -1, '與 21–120 根前的重疊度：低 = 已脫離舊區間')
    # ── 對照（已知效應，用來判斷新因子是不是換個名字的舊效應）──
    add('X_atrpct', atr / C, -1, '波動率（ATR%）：低波動效應')
    add('X_amt', np.log(AMT.rolling(20).mean()), -1, '成交額（規模／流動性代理）：小型股效應')
    add('X_price', np.log(P['RC']), -1, '未還原股價：低價股效應（V3C 的低價排序）')
    return F
