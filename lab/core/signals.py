# 訊號：T_ld、可交易池、X1 候選與排序、V4.1 事件與排序 → lab/cache/sig/<名稱>/
# 公式依 lab/finlab/fl_small_report.py 第 93–156 行；對照測試見 lab/core/test_signals.py。
# 用法：python lab/core/signals.py [compat|fix ...]
#       from lab.core.signals import compute, MODES
#
# 模式（L-113）：
#   compat：相容模式。偏誤 ① 排除 2016 起不再存在的股票、② 2016/7 前股利不還原、③ 停牌/下市以最後價凍結（③ 在 sim.py）；
#           成交額＝未還原收盤 × 股數；產業＝fl_small.py 寫死名單。
#   fix   ：修正模式。三個偏誤全關；成交額＝成交金額 value；產業＝主檔現況產業別（已知偏誤：現況分類）。
# 規則開關（預設＝定版規則；關掉＝拿掉該條件，供條件減法與高原檢查）：見 RULES。
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel, peak_mem_mb
from feat import load_feat

OUT = LAB / 'cache' / 'sig'
MODES = {
    'compat': dict(bias1=True, bias2=True, bias3=True, amt='AMT20', ind='pk'),
    'fix':    dict(bias1=False, bias2=False, bias3=False, amt='AMT20V', ind='master'),
}
RULES = dict(
    # 可交易池與強勢池
    amt_q=0.20,        # 20 日平均成交額 ≥ 全市場此分位
    pool_q=0.90,       # T_ld 池內前 10%
    floor_q=0.05,      # ATR% 下限：當日全市場第 5 百分位
    # X1
    x_hot=True, hot_q=0.80,                    # 剔過熱：20 日漲幅池內前 20%
    x_steep=True, steep_up=0.05, steep_q=0.80, # 剔最陡：S60 ≥ 5% 者中前 20%
    x_rank='both',                             # both＝½T_ld＋½ΔT_ld10；tld；dtld；random
    # V4.1
    v_ind=True,        # 產業過濾：所屬產業 T_ld 中位數排名前 50%
    v_ev1=True, v_ev2=True,                    # 布林收斂突破、整理突破 20 日高
    v_vol=True,        # 波動條件
    v_thrust=True,     # 廣度爆發免除
    v_iakf=True,       # IAKF 風控
    v_deep=True, deep_thr=0.236,               # 剔深回撤
    v_buyok_peek=True, # 參考實作：次日開盤漲停（或無開盤價）的事件不進候選。用到次日資訊（輕微前視），待登記是否改掉
    ma=200,            # 大盤均線
)


def prk(x, m):
    """當日橫斷面百分位排名（只在 m 為真的格子，平均法處理同值）。"""
    return pd.DataFrame(np.where(m, x, np.nan)).rank(axis=1, pct=True).values


def rank_desc(sc, m):
    return pd.DataFrame(np.where(m, sc, np.nan)).rank(axis=1, ascending=False, method='first').values


def compute(mode='compat', P=None, F=None, **sw):
    R = {**RULES, **sw}; md = MODES[mode] if isinstance(mode, str) else mode
    P = P or load_panel(); M = P['meta']; dates = P['dates']
    F = F or load_feat('AF_C' if md['bias2'] else 'AF')
    keep = ~M['bias1_exclude'].to_numpy() if md['bias1'] else np.ones(len(M), bool)
    cols = np.flatnonzero(keep)
    g = lambda a: np.asarray(a)[:, cols]
    VV = g(F['VALID']); Cn = g(F['CA']).astype(np.float32); ATR = g(F['ATR']); NUM = g(F['NUM'])
    AMT = g(F[md['amt']]); R20 = g(F['R20']); S60 = g(F['S60']); RRT = g(F['RRT'])
    EV1, EV2, OLD = g(F['EV1']), g(F['EV2']), g(F['OLD']); LAG = g(F['LAG20'])
    T_, N_ = Cn.shape
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()

    # ---- T_ld：(EMA12 − EMA200) ÷ max(ATR, 收盤 × 當日全市場 ATR% 第 5 百分位)
    floor = np.nanquantile(np.where(VV, ATR / Cn, np.nan), R['floor_q'], axis=1).astype(np.float32)
    DEN = np.maximum(ATR, Cn * floor[:, None]); Tr = NUM / DEN
    DP = np.full((T_, N_), np.nan, np.float32); ok = LAG >= 0
    DP[ok] = DEN[LAG[ok], np.nonzero(ok)[1]]                   # 同一檔股票往前第 20 個有效日的分母
    atr_ok = DEN < DP
    # ---- 可交易池：上市滿 250 有效日、20 日平均成交額 ≥ 當日第 20 百分位
    amt_q = np.nanquantile(np.where(VV & OLD, AMT, np.nan), R['amt_q'], axis=1)
    U = VV & OLD & (AMT >= amt_q[:, None])

    # ---- X1
    pool = U & (np.nan_to_num(prk(Tr, U)) >= R['pool_q'])
    hot = np.nan_to_num(prk(R20, pool)) > R['hot_q']
    up5 = pool & (np.nan_to_num(S60, nan=-1.0) >= R['steep_up'])
    steep = np.nan_to_num(prk(S60, up5)) > R['steep_q']
    TL = pd.DataFrame(Tr)
    dT = (TL - TL.shift(10)).values                             # 10 個市場交易日前（不是個股自身交易日，同參考實作）
    if R['x_rank'] == 'both': SCX = 0.5 * np.nan_to_num(prk(Tr, pool)) + 0.5 * np.nan_to_num(prk(dT, pool))
    elif R['x_rank'] == 'tld': SCX = np.nan_to_num(prk(Tr, pool))
    elif R['x_rank'] == 'dtld': SCX = np.nan_to_num(prk(dT, pool))
    elif R['x_rank'] == 'random': SCX = np.random.default_rng(0).random((T_, N_))
    else: raise ValueError(R['x_rank'])
    candX = pool & ~(hot if R['x_hot'] else False) & ~(steep if R['x_steep'] else False)
    RKX = rank_desc(SCX, candX)
    below = (ix < ix.rolling(R['ma']).mean()).values

    # ---- V4.1
    if md['ind'] == 'pk': ind = M['ind_pk'].to_numpy()[cols]
    else: ind = M['industry'].to_numpy()[cols]
    ind = pd.Series(ind).where(pd.notna(ind), None)
    score = np.where(U, Tr, np.nan)
    if R['v_ind']:
        kn = ind.notna().to_numpy()
        gm = pd.DataFrame(score[:, kn]).T.groupby(ind[kn].to_numpy()).transform('median').T.to_numpy()
        G = np.full((T_, N_), np.nan); G[:, kn] = gm               # 每檔股票所屬產業的 T_ld 中位數（無產業＝空）
        okg = (prk(G, U) >= 0.5) | np.isnan(G)                    # 無產業者不受產業過濾
        score = np.where(okg, score, np.nan)
    strong = pd.DataFrame(score).rank(axis=1, pct=True).values >= R['pool_q']
    RC = g(P['C']).astype(float); ROP = g(P['O']).astype(float)
    if R['v_buyok_peek']:
        lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10) - 0.005
        nxt = np.vstack([ROP[1:], np.full((1, N_), np.nan)]) / RC - 1
        lim_n = np.r_[lim[1:], np.nan]
        Oadj_n = np.vstack([np.asarray(g(F['OA']))[1:], np.full((1, N_), np.nan)])
        with np.errstate(invalid='ignore'):
            buy_ok = ~(nxt >= lim_n[:, None]) & np.isfinite(Oadj_n)
    else:
        buy_ok = np.ones((T_, N_), bool)
    bear200 = pd.Series(below, index=dates)
    ixr = ix.pct_change(fill_method=None)
    mv = ixr.rolling(20).std(); mkt_vol_down = (mv < mv.shift(20)).values
    Cf = pd.DataFrame(Cn.astype(float)).ffill()                  # 參考實作 C.pct_change()（pandas 2 預設先向前填補）
    Ush = np.vstack([np.zeros((1, N_), bool), U[:-1]])
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # 全空的列（nanmean）→ 0
        ew = pd.Series(np.nanmean(np.where(Ush, (Cf / Cf.shift(1) - 1).values, np.nan), axis=1), index=dates).fillna(0)
    ratio_ew = (1 + ew).cumprod() / ix
    thrust = (ix.pct_change(10, fill_method=None) >= 0.05) & (ratio_ew.pct_change(10, fill_method=None) >= 0.02)
    thrust_win = thrust.rolling(20, min_periods=1).max().astype(bool)
    if not R['v_thrust']: thrust_win = pd.Series(False, index=dates)
    bear_mode = bear200 & ~thrust_win; MD = bear_mode.values
    if R['v_iakf']:
        OFF = pd.Series(F['market']['IAKF_DN'].reindex(dates).fillna(False).values, index=dates).rolling(5, min_periods=1).max().astype(bool).values
    else:
        OFF = np.zeros(T_, bool)
    if R['v_vol']:
        allow = (~MD[:, None] & (atr_ok | thrust_win.values[:, None])) | (MD[:, None] & mkt_vol_down[:, None])
    else:
        allow = np.ones((T_, N_), bool)
    deep = np.nan_to_num(RRT, nan=0.0) >= R['deep_thr']
    ev = (EV1 if R['v_ev1'] else False) | (EV2 if R['v_ev2'] else False)
    sig0 = strong & ev & allow & ~OFF[:, None] & buy_ok
    signal = sig0 & ~deep if R['v_deep'] else sig0
    RKV = rank_desc(score, signal)

    def full(a, fill=np.nan, dt=np.float32):
        o = np.full((T_, len(M)), fill, dt); o[:, cols] = a; return o
    daily = pd.DataFrame(dict(
        n_valid=VV.sum(1), n_U=U.sum(1), n_pool=pool.sum(1), n_hot=(pool & hot).sum(1), n_up5=up5.sum(1),
        n_steep=(pool & steep).sum(1), n_candX=candX.sum(1), n_strongV=strong.sum(1), n_sig0=sig0.sum(1),
        n_deep=(sig0 & deep).sum(1), n_signal=signal.sum(1), below=below, bear_mode=MD, off=OFF,
        thrust_win=thrust_win.values, floor=floor, amt_q=amt_q), index=dates)
    return dict(RKX=full(RKX), RKV=full(RKV), ATRP=full(ATR / Cn), daily=daily, rules=R, mode=md,
                _dbg=dict(Tr=Tr, U=U, pool=pool, hot=hot, steep=steep, strong=strong, sig0=sig0, signal=signal, cols=cols))


def build(name='compat', mode=None, P=None, F=None, **sw):
    """name：快取資料夾名稱；mode：MODES 的鍵或自訂 dict（預設＝name）；sw：規則開關。"""
    t0 = time.time(); r = compute(mode or name, P=P, F=F, **sw)
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    for k in ('RKX', 'RKV', 'ATRP'): np.save(d / f'{k}.npy', r[k])
    r['daily'].reset_index(names='date').to_parquet(d / 'daily.parquet', index=False)
    info = dict(name=name, flags=r['mode'], rules=r['rules'], seconds=round(time.time() - t0), peak_mb=round(peak_mem_mb()))
    (d / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return info, r['daily']


def load_sig(name='compat'):
    d = OUT / name
    out = {k: np.load(d / f'{k}.npy', mmap_mode='r') for k in ('RKX', 'RKV', 'ATRP')}
    out['daily'] = pd.read_parquet(d / 'daily.parquet').set_index('date')
    out['info'] = json.loads((d / 'info.json').read_text(encoding='utf-8'))
    return out


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    for nm in (sys.argv[1:] or ['compat', 'fix']):
        info, daily = build(nm)
        print(nm, info['seconds'], 's', info['peak_mb'], 'MB', flush=True)
