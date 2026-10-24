"""第四批：結構類（全部無尺度：價格帶用 ATR 單位、量用占比、位置用比例）
A. 籌碼分布：過去 K 棒的成交量落在哪些價位
   - 上方壓力：成交量落在 [近5日高, +k×ATR] 的占比（套牢盤）
   - 下方支撐：成交量落在 [收盤 − k×ATR, 收盤] 的占比（支撐盤）
   - 獲利盤比例：過去成交量中，價格低於今天收盤的比例（持有者多數賺錢 → 賣壓小）
B. 型態：收斂、區間位置、突破距離、破前低、重疊密度
方向都在看數據前宣告。"""
import numpy as np, pandas as pd
from factors import base
from factors3 import tld_floor


def _lag(a, k):
    out = np.full_like(a, np.nan); out[k:] = a[:-k]; return out


def profile(P, a=21, b=250, k_atr=3.0, half_life=None):
    """回傳 dict：上方壓力、下方支撐、獲利盤比例（成交量占比），用 t−b～t−a 的 K 棒
    K 棒的量依價格區間均勻分攤：落在某價格帶的量 = V × (區間與價格帶的重疊長度 / K 棒長度)"""
    H, L, C, V = (P[k].values.astype('float64') for k in ('H', 'L', 'C', 'V'))
    atr = base(P)['atr'].values
    hi5 = pd.DataFrame(H).rolling(5).max().values
    up_lo, up_hi = hi5, hi5 + k_atr * atr
    dn_lo, dn_hi = C - k_atr * atr, C
    tot = np.zeros_like(C); ovh = np.zeros_like(C); sup = np.zeros_like(C); prof = np.zeros_like(C)
    for k in range(a, b + 1):
        h, l, v = _lag(H, k), _lag(L, k), _lag(V, k)
        w = 0.5 ** (k / half_life) if half_life else 1.0
        v = np.nan_to_num(v) * w
        rng = np.maximum(h - l, 1e-9 * np.nan_to_num(h, nan=1.0))
        f_up = np.clip(np.minimum(h, up_hi) - np.maximum(l, up_lo), 0, None) / rng
        f_dn = np.clip(np.minimum(h, dn_hi) - np.maximum(l, dn_lo), 0, None) / rng
        f_pf = np.clip((C - l) / rng, 0, 1)
        ok = ~np.isnan(h)
        tot += np.where(ok, v, 0)
        ovh += np.where(ok, v * np.nan_to_num(f_up), 0)
        sup += np.where(ok, v * np.nan_to_num(f_dn), 0)
        prof += np.where(ok, v * np.nan_to_num(f_pf), 0)
    age = P['C'].notna().cumsum().values
    good = (age >= b) & (tot > 0)
    mk = lambda x: pd.DataFrame(np.where(good, x / np.where(tot > 0, tot, np.nan), np.nan), index=P['C'].index, columns=P['C'].columns)
    return dict(ovh=mk(ovh), sup=mk(sup), prof=mk(prof))


def build4(P, daily=None):
    C, H, L = P['C'], P['H'], P['L']
    B = base(P); atr = B['atr']
    F = {}
    def add(name, X, sign, why): F[name] = (X.astype('float32'), sign, why)
    add('T_ldf', tld_floor(P), +1, '基準')
    # ── A. 籌碼分布 ──
    p1 = profile(P, 21, 250, 3.0)
    add('A_ovh250', p1['ovh'], -1, '上方壓力：近 1 年（跳過最近 1 個月）成交量落在 [近5日高, +3ATR] 的占比')
    add('A_sup250', p1['sup'], +1, '下方支撐：成交量落在 [收盤−3ATR, 收盤] 的占比')
    add('A_prof250', p1['prof'], +1, '獲利盤比例：近 1 年成交量中價格低於今天收盤的比例')
    add('A_supovh', np.log((p1['sup'] + 0.01) / (p1['ovh'] + 0.01)), +1, '支撐／壓力比（log）')
    p2 = profile(P, 1, 60, 3.0)
    add('A_prof60', p2['prof'], +1, '近 60 日獲利盤比例（短期籌碼）')
    add('A_ovh60', p2['ovh'], -1, '近 60 日上方壓力')
    p3 = profile(P, 21, 500, 3.0, half_life=125)
    add('A_ovh500w', p3['ovh'], -1, '上方壓力，近 2 年、半衰期半年（越舊的套牢越不重要）')
    add('A_prof500w', p3['prof'], +1, '獲利盤比例，近 2 年、半衰期半年')
    p4 = profile(P, 21, 250, 1.5)
    add('A_ovh250n', p4['ovh'], -1, '上方壓力（窄帶 1.5ATR）')
    # ── B. 型態 ──
    rng20 = H.rolling(20).max() - L.rolling(20).min()
    add('B_tight20', rng20 / atr, -1, '近 20 日高低區間 ÷ ATR：越小 = 越收斂（底部打得越緊）')
    add('B_vcp', atr.rolling(20).mean() / atr.rolling(120).mean(), -1, 'ATR 收縮：20 日均 ATR ÷ 120 日均 ATR')
    hi250, lo250 = H.rolling(250).max(), L.rolling(250).min()
    add('B_pos250', (C - lo250) / (hi250 - lo250), +1, '在一年高低區間的位置（0 = 低點，1 = 高點）')
    add('B_bo60', (C - H.shift(5).rolling(55).max()) / atr, +1, '收盤高於前 5～60 日最高價幾個 ATR：突破程度')
    add('B_dd250', (C - hi250) / atr, +1, '距一年高點幾個 ATR（負值，越接近 0 越好）')
    lo_ratio = L.shift(20).rolling(230).min() / L.shift(250).rolling(250).min()
    add('B_loratio', lo_ratio, -1, '破前低後打底（近一年低點 ÷ 前一年低點）')
    rec = (C - L.rolling(120).min()) / (H.rolling(250).max() - L.rolling(120).min())
    add('B_recover', rec, +1, '從近半年低點回升到一年高點的比例')
    if daily is not None:
        add('B_near5', daily['near5'], +1, '近 20 根重疊度（近期收斂）')
        add('B_far5', daily['far5'], -1, '與 21～120 根前的重疊度（已脫離舊區間）')
        add('B_trans20', daily['trans20'], +1, '近 20 日出現過 C→B→A 結構轉變（0/1）')
    return F
