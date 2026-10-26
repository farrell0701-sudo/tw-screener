"""EMA(f) 與 MA(s) 交叉：事件研究、單獨進出場、搭配收斂突破事件（時間窗內先後發生，以後到者進場）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
EV = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
anyev = (EV['布林：收斂後首次突破上軌'] | EV['布林：收斂後帶寬張開 + 收上軌'] | EV['整理後突破 20 日高']).reindex_like(C).fillna(False)
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
strong = (cur.rank(axis=1, pct=True) >= 0.9) & ~ldn5
def cross(f, s):
    e = C.ewm(span=f, adjust=False).mean(); m = C.rolling(s).mean(); d = e - m
    return (d > 0) & (d.shift(1) <= 0), (d < 0) & (d.shift(1) >= 0), d > 0
def recent(m, w): return m.astype(float).rolling(w, min_periods=1).max() > 0
def eng(entry, label, exit_sig=None, H=None, n=10, ladder=None):
    """entry：當天收盤訊號（強勢股內）；exit_sig：死叉出場（隔日開盤）；H：固定持有天數"""
    if ladder:
        rs = [eng(entry, None, exit_sig, h, n) for h in ladder]
        r = sum(x.pct_change() for x in rs) / len(rs); return report(r, label)
    S = strong & entry.fillna(False)
    RKs = cur.where(S & BUYdf).rank(axis=1, ascending=False, method='first').values
    XS = None if exit_sig is None else exit_sig.fillna(False).values
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan); nt = 0
    for t in range(d0, len(dates)):
        keep = set()
        for j in sell_next:
            if j not in pos: continue
            if lockdn[t, j]: keep.add(j); continue
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX); nt += 1
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or lockup[t, j]: continue
                amt = min(val / n, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, t]; cash -= amt
        sell_next, buy_next = keep, []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        for j, p in pos.items():
            if j in sell_next: continue
            held = t - p[2] + 1
            if H and held >= H: sell_next.add(j); continue
            if XS is not None and XS[t, j]: sell_next.add(j); continue
            if np.isnan(Cv[t, j]) and held > 60: sell_next.add(j)
        free = n - len(pos) + len(sell_next & set(pos))
        if free > 0:
            cand = np.where(~np.isnan(RKs[t]))[0]
            buy_next = [j for j in cand[np.argsort(RKs[t, cand])] if j not in pos][:free]
    s = pd.Series(eq, index=dates).dropna()
    if label is None: return s
    report(s.pct_change(), label, nt); return s
rows = []
def report(r, label, nt=None):
    r = r.dropna(); e = (1 + r).cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    yr = e.resample('YE').last().pct_change(); yr.iloc[0] = e.resample('YE').last().iloc[0] - 1
    o = dict(版本=label, 年化=f'{e.iloc[-1] ** (1 / y) - 1:.1%}', 回撤=f'{(e / e.cummax() - 1).min():.1%}', Sharpe=round(r.mean() / r.std() * np.sqrt(252), 2),
             最差年=f'{yr.min():.1%}', 每年交易=round(nt / y) if nt else '')
    rows.append(o); print(o, flush=True); return r
if __name__ == '__main__':
    # 1. 事件研究：強勢股內金叉
    ent = O.shift(-1); Cf = C.ffill(); dd0 = pd.Timestamp('2016-02-01')
    lu_next = (O.shift(-1) / Cf - 1 >= 0.095)
    base_m = (strong & ~lu_next & ent.notna()).loc[dd0:]
    er = []
    for f, s_ in ((10, 20), (15, 20), (20, 30)):
        up, dn, _ = cross(f, s_)
        for lab, m_ in ((f'EMA{f}/MA{s_} 金叉', up), (f'EMA{f}/MA{s_} 金叉＋10 日內有收斂突破事件', up & recent(anyev, 10))):
            o = {'事件': lab}
            for h in (5, 20, 60):
                R = (Cf.shift(-h) / ent - 1).loc[dd0:]; m = base_m & m_.loc[dd0:].fillna(False) & R.notna()
                ex = R.sub(R.where(base_m).mean(1), axis=0).where(m).stack().dropna()
                yr = ex.groupby(ex.index.get_level_values(0).year).mean()
                o[f'{h}日超額'] = f'{ex.mean():+.2%}'; o[f'{h}日正年'] = f'{(yr > 0).sum()}/{len(yr)}'
            o['占強勢股日'] = f'{m.sum().sum() / base_m.sum().sum():.1%}'; er.append(o)
    pd.set_option('display.width', 260); print(pd.DataFrame(er).to_string(index=False), flush=True)
    # 2～4. 策略
    LAD = (15, 20, 25)
    for f, s_ in ((15, 20), (10, 20), (20, 30)):
        up, dn, above = cross(f, s_)
        tag = f'{f}/{s_}'
        eng(up, f'[{tag}] 單獨：金叉買、死叉賣', exit_sig=dn)
        for W in ((10,) if tag != '15/20' else (5, 10, 20)):
            both = (up & recent(anyev, W)) | (anyev & recent(up, W) & above)     # 事件後 W 日內金叉，或金叉後 W 日內出現事件（且仍在金叉狀態）
            eng(both, f'[{tag}] 事件＋金叉（{W} 日內，後到者進場），死叉賣', exit_sig=dn)
            eng(both, f'[{tag}] 事件＋金叉（{W} 日內），分批 15/20/25 日出場', ladder=LAD)
        eng(anyev, f'[{tag}] 短線事件版，出場改死叉', exit_sig=dn)
        eng(anyev, f'[{tag}] 短線事件版，死叉或滿 25 日先到者', exit_sig=dn, H=25)
    eng(anyev, '對照：短線事件版（分批 15/20/25）', ladder=LAD)
    D = pd.DataFrame(rows); print(D.to_string(index=False)); D.to_csv('/home/claude/bt/fac/emacross.csv', index=False, encoding='utf-8-sig')
