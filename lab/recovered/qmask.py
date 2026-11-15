import os, numpy as np, pandas as pd
def build_q(dates, cols):
    T_, N_ = len(dates), len(cols); Q = np.zeros((T_, N_), bool); cov = np.zeros((T_, N_), bool)
    lag = {3: (5, 15, 0), 6: (8, 14, 0), 9: (11, 14, 0), 12: (3, 31, 1)}
    for k, c in enumerate(cols):
        fi, fb = f'/home/claude/fund/is_{c}.csv', f'/home/claude/fund/bs_{c}.csv'
        if not (os.path.exists(fi) and os.path.getsize(fi) > 50 and os.path.exists(fb) and os.path.getsize(fb) > 50): continue
        i = pd.read_csv(fi, usecols=['date', 'type', 'value']); b = pd.read_csv(fb, usecols=['date', 'type', 'value'])
        ni = i[i.type == 'IncomeAfterTaxes'].drop_duplicates('date', keep='last').set_index('date').value
        bb = b[b.type.isin(['EquityAttributableToOwnersOfParent', 'CurrentAssets', 'CurrentLiabilities'])].drop_duplicates(['date', 'type'], keep='last').pivot(index='date', columns='type', values='value')
        q = pd.concat([ni.rename('NI'), bb], axis=1).sort_index(); q.index = pd.to_datetime(q.index)
        if len(q) < 12 or 'EquityAttributableToOwnersOfParent' not in q: continue
        q = q.asfreq('QE')
        roe = q.NI.rolling(12, min_periods=12).sum() / 3 / q.EquityAttributableToOwnersOfParent.rolling(12, min_periods=12).mean()
        y = q.NI.rolling(4, min_periods=4).sum(); pos = (y > 0) & (y.shift(4) > 0) & (y.shift(8) > 0)
        cr = q.get('CurrentAssets') / q.get('CurrentLiabilities') if 'CurrentAssets' in q and 'CurrentLiabilities' in q else pd.Series(np.nan, index=q.index)
        ok = (roe > 0.12) & pos & (cr > 1.2); have = roe.notna()
        av = [pd.Timestamp(d.year + lag[d.month][2], lag[d.month][0], lag[d.month][1]) for d in q.index]
        s = pd.DataFrame({'ok': ok.values, 'have': have.values}, index=av); s = s[~s.index.duplicated(keep='last')].sort_index()
        s = s.reindex(s.index.union(dates)).ffill().reindex(dates)
        Q[:, k] = s.ok.fillna(False).astype(bool).values; cov[:, k] = s.have.fillna(False).astype(bool).values
    return Q, cov
