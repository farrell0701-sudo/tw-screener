"""第六批：基本面（價值、獲利、成長、規模），只用當時已公布的資料（build_fund.py）
方向在看數據前宣告"""
import numpy as np, pandas as pd
from factors3 import tld_floor


def build6(P, W):
    C, RC = P['C'], P['RC']
    mcap = RC * W['SH'] / 10                                # 普通股股本 ÷ 面額 10 元 = 股數
    F = {}
    def add(name, X, sign, why): F[name] = (X.astype('float32'), sign, why)
    add('T_ldf', tld_floor(P), +1, '基準')
    add('F_EP', 1 / W['PER'].where(W['PER'] > 0), +1, '盈餘殖利率 1/本益比（證交所每日；虧損股無值）')
    add('F_EPfin', W['NI_TTM'] / mcap, +1, '近四季歸母淨利 ÷ 市值（含虧損股）')
    add('F_BP', 1 / W['PBR'].where(W['PBR'] > 0), +1, '淨值市價比 1/股價淨值比')
    add('F_DY', W['DY'], +1, '現金殖利率')
    add('F_ROE', W['NI_TTM'] / W['EQ'].where(W['EQ'] > 0), +1, 'ROE（近四季淨利 ÷ 最新歸母權益）')
    add('F_GM', W['GM_TTM'], +1, '毛利率（近四季）')
    add('F_OPM', W['OPM_TTM'], +1, '營業利益率（近四季）')
    add('F_REVYOY', np.log(W['REV3'] / W['REV3_LY'].where(W['REV3_LY'] > 0)), +1, '近 3 個月營收年增率（log）')
    add('F_REVMOM', W['REV3'] / (W['REV12'].where(W['REV12'] > 0) / 4), +1, '近 3 個月營收 ÷ 近 12 個月月均×3：營收加速')
    add('F_SIZE', np.log(mcap), -1, '市值（log）：小型股效應')
    return F
