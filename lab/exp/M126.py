# M-126：外資持股因子（依 M-119；奇數代碼）。定義見 registry_M.md M-126。
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'core'))
from M125 import monthly_from_daily, report
from chips import _staged


def fx_factors(dates):
    df = _staged('TaiwanStockShareholding', ['date', 'stock_id', 'ForeignInvestmentSharesRatio'])
    def fn(W):
        r = W['ForeignInvestmentSharesRatio']
        return {'W1 外資持股 20 日變化': r - r.shift(20), 'W2 外資持股 60 日變化': r - r.shift(60), 'W3 外資持股水準': r}
    return monthly_from_daily(df, ['ForeignInvestmentSharesRatio'], dates, fn)


def main():
    return dict(summary=report('# M-126 結果：外資持股因子（奇數代碼；F 計數 16）\n', fx_factors), headline='外資持股因子評估完成', n_variants=0)
