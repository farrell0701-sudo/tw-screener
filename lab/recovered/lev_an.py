"""只在多頭開槓桿：2005–今（2005–2016/01 舊股價、2016/02 起新資料，各 20 路徑逐條拼接）
融資模型：多頭期間借款 =(L−1)×淨值，每 20 個交易日或多頭開始時重設；空頭當天收盤後還清（隔日 1 倍）
利息年 6.45%（借款部分）；維持率 = 資產/借款 < 130% → 追繳，當天強制還清，等下一次重設才再借"""
import numpy as np, pandas as pd
O, N = pd.read_pickle('/home/claude/old/lev_old.pkl'), pd.read_pickle('/home/claude/old/lev_new.pkl')
bear = pd.concat([O['bear'].loc[:'2016-01-29'], N['bear'].loc['2016-02-01':]])
paths = [pd.concat([a.loc['2005-01-03':'2016-01-29'], b.loc['2016-02-01':]]) for a, b in zip(O['paths'], N['paths'])]
RATE = 0.0645 / 252


def lever(r, L, bull_only=True, reset=20):
    br = bear.reindex(r.index).fillna(False).shift(1).fillna(False).values      # 昨天收盤判斷
    eq = 1.0; loan = 0.0; out = np.empty(len(r)); calls = 0; k = 0
    for i, x in enumerate(r.values):
        want = (L > 1) and (not bull_only or not br[i])
        if want and (loan == 0 or k >= reset):
            loan = (L - 1) * eq; k = 0
        if not want: loan = 0.0
        asset = eq + loan
        asset *= 1 + x; eq_new = asset - loan - loan * RATE
        if loan > 0 and asset / loan < 1.3:            # 追繳：強制還款
            calls += 1; loan = 0.0; k = -10 ** 9 if False else 0
        eq = eq_new; out[i] = eq; k += 1
    return pd.Series(out, index=r.index), calls


def stat(e):
    r = e.pct_change().fillna(0); y = (e.index[-1] - e.index[0]).days / 365.25
    yr = e.resample('YE').last(); yr = yr.pct_change().fillna(yr.iloc[0] / e.iloc[0] - 1)
    return dict(年化=(e.iloc[-1] / e.iloc[0]) ** (1 / y) - 1, 回撤=(e / e.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252), 最差年=yr.min()), yr


V = {'1 倍（不開槓桿）': (1, True), '全程 2.5 倍': (2.5, False), '多頭 1.5 倍、空頭 1 倍': (1.5, True),
     '多頭 2 倍、空頭 1 倍': (2.0, True), '多頭 2.5 倍、空頭 1 倍': (2.5, True)}
rows, YR = [], {}
for k, (L, bo) in V.items():
    E, C_ = zip(*[lever(p, L, bo) for p in paths])
    S = pd.DataFrame([stat(e)[0] for e in E]).median()
    row = dict(版本=k, 年化=f'{S.年化:.1%}', 回撤=f'{S.回撤:.1%}', Sharpe=f'{S.Sharpe:.2f}', 最差年=f'{S.最差年:.1%}', 追繳次數=f'{np.median(C_):.0f}')
    for a, b in (('2005', '2015'), ('2016', '2026')):
        s_ = pd.DataFrame([stat(e.loc[a:b])[0] for e in E]).median(); row[f'{a}-{b[2:]}'] = f'{s_.年化:.0%}/{s_.回撤:.0%}'
    eqm = pd.concat([e.pct_change().fillna(0) for e in E], axis=1).mean(axis=1)
    row['100萬變成(萬)'] = f'{np.median([e.iloc[-1] for e in E]) * 100:,.0f}'
    rows.append(row); YR[k] = stat((1 + eqm).cumprod())[1]
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False))
Y = pd.DataFrame(YR); Y.index = Y.index.year; print((Y * 100).round(1).to_string())
print('空頭天數占比', f'{bear.loc["2005":].mean():.0%}')
