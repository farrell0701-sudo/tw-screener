"""第五批：量價類（全部和個股自己的歷史比，或用比例；方向在看數據前宣告）
邏輯分組
  注意力：成交量相對自己過去放大（資金開始關注）
  吸籌：上漲日的量多於下跌日、收盤常收在高檔（主力買進）
  爆量：單日巨量（可能是突破，也可能是出貨）
  量縮：價格強但量縮（籌碼安定、賣壓小）
  流動性：Amihud 非流動性（低流動性溢酬）"""
import numpy as np, pandas as pd
from factors3 import tld_floor


def build5(P):
    C, H, L, V, AMT = P['C'], P['H'], P['L'], P['V'], P['AMT']
    r = C / C.shift(1) - 1
    lv = np.log(V.where(V > 0))
    F = {}
    def add(name, X, sign, why): F[name] = (X.astype('float32'), sign, why)
    add('T_ldf', tld_floor(P), +1, '基準')
    # 注意力
    add('V_ratio20', np.log(V.rolling(20).mean() / V.rolling(120).mean()), +1, '20 日均量 ÷ 120 日均量（log）：關注度上升')
    add('V_ratio5', np.log(V.rolling(5).mean() / V.rolling(60).mean()), +1, '5 日均量 ÷ 60 日均量（log）：短期放量')
    # 吸籌
    up = V.where(r > 0, 0).rolling(60).sum(); dn = V.where(r < 0, 0).rolling(60).sum()
    add('V_updn60', np.log((up + 1) / (dn + 1)), +1, '60 日上漲日成交量 ÷ 下跌日成交量（log）：買盤主導')
    up20 = V.where(r > 0, 0).rolling(20).sum(); dn20 = V.where(r < 0, 0).rolling(20).sum()
    add('V_updn20', np.log((up20 + 1) / (dn20 + 1)), +1, '20 日上漲日量 ÷ 下跌日量')
    clv = ((C - L) - (H - C)) / (H - L).where(H > L)
    add('V_cmf20', (clv * V).rolling(20).sum() / V.rolling(20).sum(), +1, 'Chaikin 資金流（20 日）：量加權的收盤位置，收在高檔＝買盤')
    add('V_clv20', clv.rolling(20).mean(), +1, '20 日平均收盤位置（不加權，對照）')
    obv = (np.sign(r).fillna(0) * V).cumsum()
    add('V_obv60', (obv - obv.shift(60)) / V.rolling(60).sum(), +1, 'OBV 60 日變化 ÷ 60 日總量：淨買進量占比')
    # 量價相關
    dlv = lv.diff()
    add('V_pvcorr60', r.rolling(60).corr(dlv), +1, '60 日報酬與量變化的相關：價漲量增、價跌量縮')
    # 爆量
    add('V_spike20', np.log(V.rolling(20).max() / V.rolling(120).median()), -1, '近 20 日最大量 ÷ 120 日中位量（log）：爆量後易出貨')
    # 量縮
    add('V_dry', np.log(V.rolling(5).mean() / V.rolling(20).max()), -1, '5 日均量 ÷ 20 日最大量（log）：量縮整理')
    # 流動性
    amihud = (r.abs() / AMT.where(AMT > 0)).rolling(20).mean()
    add('V_amihud', np.log(amihud * 1e9), +1, 'Amihud 非流動性（log）：流動性差的股票有溢酬')
    add('V_amtgrow', np.log(AMT.rolling(20).mean() / AMT.rolling(250).mean()), +1, '20 日成交額 ÷ 250 日成交額（log）：資金流入規模')
    return F
