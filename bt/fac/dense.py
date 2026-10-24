"""密集重疊區突破（通用版，不限 C→B→A）
密集區：最近 W 根 K 棒中，至少一半 K 棒都覆蓋的價格區間
        ＝ [W 根低點的中位數, W 根高點的中位數]（數學上等價：價位 p 被 ≥ W/2 根 K 棒覆蓋）
整理條件：W 根總區間 ÷ ATR 在當天全市場的百分位 ≤ q（越小 = 越像整理；q=1 不限制）
訊號：今天收盤 > 昨天為止的密集區上緣（或箱頂 = W 根最高價），且昨天收盤還沒越過
全部只用到當天以前的資料。"""
import numpy as np, pandas as pd


def dense_signals(P, atr, U, W=20, q=1.0, edge='dense'):
    H, L, C = P['H'], P['L'], P['C']
    medH = H.rolling(W, min_periods=W).median().shift(1)        # 用到昨天為止的 W 根
    maxH = H.rolling(W, min_periods=W).max().shift(1)
    rng = (H.rolling(W).max() - L.rolling(W).min()).shift(1) / atr.shift(1)
    top = medH if edge == 'dense' else maxH
    cross = (C > top) & (C.shift(1) <= top.shift(1))
    if q < 1:
        rp = rng.where(U).rank(axis=1, pct=True)
        cross &= rp.shift(0) <= q
    return cross.fillna(False) & U
