"""抗跌在空頭的表現：依當天多空（指數 EMA60 斜率）拆開每日報酬；另看幾段主要下跌；再測「空頭才改挑抗跌」"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
src = open('/home/claude/bt/fac/port_dd.py').read().split("for K in (20, 30, 40):")[0]
exec(src)
K = 20; pool = rk <= K
xp = F3['R_downday'][0].where(U).rank(axis=1, pct=True)
S_now = base_sc.where(pool)
S_dd = (xp + base_sc.rank(axis=1, pct=True) * 1e-6).where(pool)
bull_df = pd.DataFrame(np.repeat(BULL[:, None], C.shape[1], 1), index=dates, columns=C.columns)
S_sw = S_now.rank(axis=1, pct=True).where(bull_df, S_dd.rank(axis=1, pct=True))   # 換股日是多頭用 T_ldf，空頭用抗跌
V = {'T_ldf（現行）': S_now, '抗跌最高': S_dd, '空頭才改挑抗跌': S_sw}
reg = pd.Series(BULL, index=dates).shift(1)          # 用前一天收盤判斷的多空
EP = {'2018-10 美中貿易戰': ('2018-09-28', '2018-12-31'), '2020-02 疫情崩跌': ('2020-01-20', '2020-03-19'),
      '2022 升息空頭': ('2022-01-17', '2022-10-25'), '2024-07 AI 回檔': ('2024-07-11', '2024-08-05'), '2025-04 關稅': ('2025-03-26', '2025-04-09')}
tx = P['taiex']
rows = []
for name, sc in V.items():
    globals()['Sv'] = sc.values
    curves = [run(off, False, 4, 3)[0] for off in range(20)]
    rec = dict(版本=name)
    rb, rr = [], []
    for s in curves:
        r = s.pct_change().dropna(); g = reg.reindex(r.index)
        rb.append((1 + r[g == True]).prod() ** (252 / max(1, (g == True).sum())) - 1)
        rr.append((1 + r[g == False]).prod() ** (252 / max(1, (g == False).sum())) - 1)
    rec['多頭日年化'] = np.median(rb) * 100; rec['空頭日年化'] = np.median(rr) * 100
    for ep, (a, b) in EP.items():
        rec[ep] = np.median([(s[a:b].iloc[-1] / s[a:b].iloc[0] - 1) * 100 for s in curves])
    st = pd.DataFrame([stats(s) for s in curves]).median()
    rec.update(全期年化=st.CAGR * 100, 回撤=st.MDD * 100, Sharpe=st.Sharpe)
    rows.append(rec); print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
r = tx.pct_change(); g = reg
print('加權指數：多頭日年化 %.1f%%  空頭日年化 %.1f%%' % (((1 + r[g == True].loc['2016-02':]).prod() ** (252 / (g.loc['2016-02':] == True).sum()) - 1) * 100,
                                               ((1 + r[g == False].loc['2016-02':]).prod() ** (252 / (g.loc['2016-02':] == False).sum()) - 1) * 100))
for ep, (a, b) in EP.items(): print(ep, '指數 %.1f%%' % ((tx[a:b].iloc[-1] / tx[a:b].iloc[0] - 1) * 100))
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_dd_bear.csv', index=False, encoding='utf-8-sig')
