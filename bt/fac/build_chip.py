"""籌碼面板（FinMind）：法人買賣超（股）、融資融券餘額（張→股）、當沖成交量（股）；當天收盤後可用"""
import sys, os; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
P = load_panel(); idx, cols = P['C'].index, P['C'].columns
D = '/home/claude/chip'
out = {k: {} for k in ('foreign', 'trust', 'dealer', 'mbal', 'sbal', 'mbuy', 'dtv')}
def rd(f):
    try: return pd.read_csv(f, parse_dates=['date']).drop_duplicates('date').set_index('date')
    except Exception: return None
for c in cols:
    x = rd(f'{D}/inst_{c}.csv')
    if x is not None and len(x):
        for k in ('foreign', 'trust', 'dealer'):
            if k in x: out[k][c] = x[k]
    x = rd(f'{D}/mg_{c}.csv')
    if x is not None and len(x):
        if 'MarginPurchaseTodayBalance' in x: out['mbal'][c] = x.MarginPurchaseTodayBalance * 1000
        if 'ShortSaleTodayBalance' in x: out['sbal'][c] = x.ShortSaleTodayBalance * 1000
        if 'MarginPurchaseBuy' in x: out['mbuy'][c] = x.MarginPurchaseBuy * 1000
    x = rd(f'{D}/dt_{c}.csv')
    if x is not None and len(x) and 'Volume' in x: out['dtv'][c] = x.Volume
W = {k: pd.DataFrame(v).reindex(index=idx, columns=cols).astype('float32') for k, v in out.items()}
pd.to_pickle(W, '/home/claude/bt/fac/chip.pkl')
d = idx[-300]
print({k: int(v.loc[d].notna().sum()) for k, v in W.items()})
V = P['V']
for k in ('foreign', 'trust', 'dtv', 'mbal'):
    r = (W[k].abs() / V).loc['2020':].stack()
    print(k, '|值| ÷ 成交量 中位數', round(float(r.median()), 4))
