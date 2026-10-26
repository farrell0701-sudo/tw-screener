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
def eng(entry, label, exit_sig=None, H=None, n=10, ladder=None, minx=0):
    """entry：當天收盤訊號（強勢股內）；exit_sig：死叉出場（隔日開盤）；H：固定持有天數"""
    if ladder:
        rs = [eng(entry, None, exit_sig, h, n, minx=minx) for h in ladder]
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
            if XS is not None and held >= minx and XS[t, j]: sell_next.add(j); continue
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
    ATR = pd.DataFrame(Av, index=C.index, columns=codes)
    def gap(f, s_): return (C.ewm(span=f, adjust=False).mean() - C.rolling(s_).mean()) / ATR
    G = {p: gap(*p) for p in ((10, 20), (15, 20), (20, 30))}
    tag = lambda p: f'{p[0]}/{p[1]}'
    eng(anyev, '對照：短線事件版（分批 15/20/25）', ladder=(15, 20, 25))
    for p, g in G.items():
        eng(anyev, f'[{tag(p)}] 原版：跌破 0（死叉狀態）或滿 25 日', exit_sig=g < 0, H=25)
    print('── 1. 緩衝區（ATR 標準化）', flush=True)
    for b in (0.2, 0.3, 0.5):
        eng(anyev, f'[15/20] 差距 < −{b} ATR 或滿 25 日', exit_sig=G[(15, 20)] < -b, H=25)
    for p in ((10, 20), (20, 30)):
        eng(anyev, f'[{tag(p)}] 差距 < −0.3 ATR 或滿 25 日', exit_sig=G[p] < -0.3, H=25)
    print('── 2. 最短持有 10 日', flush=True)
    for p, g in G.items():
        eng(anyev, f'[{tag(p)}] 第 10 日起死叉狀態出場，最多 25 日', exit_sig=g < 0, H=25, minx=10)
        eng(anyev, f'[{tag(p)}] 第 10 日起差距 < −0.3 ATR 出場，最多 25 日', exit_sig=g < -0.3, H=25, minx=10)
    print('── 3. 三組均線至少兩組同意', flush=True)
    nb0 = sum((g < 0).astype(int) for g in G.values()); nb3 = sum((g < -0.3).astype(int) for g in G.values())
    eng(anyev, '[共識] 2/3 組死叉狀態或滿 25 日', exit_sig=nb0 >= 2, H=25)
    eng(anyev, '[共識] 2/3 組差距 < −0.3 ATR 或滿 25 日', exit_sig=nb3 >= 2, H=25)
    eng(anyev, '[共識] 第 10 日起 2/3 組死叉狀態，最多 25 日', exit_sig=nb0 >= 2, H=25, minx=10)
    eng(anyev, '[共識] 第 10 日起 2/3 組差距 < −0.3 ATR，最多 25 日', exit_sig=nb3 >= 2, H=25, minx=10)
    print('── 單獨使用：緩衝區進出（不看事件）', flush=True)
    for b in (0.3, 0.5):
        g = G[(15, 20)]
        eng((g > b) & (g.shift(1) <= b), f'[15/20 單獨] 差距上穿 +{b} ATR 買、跌破 −{b} ATR 賣', exit_sig=g < -b)
    D = pd.DataFrame(rows); pd.set_option('display.width', 250); print(D.to_string(index=False))
    D.to_csv('/home/claude/bt/fac/emacross2.csv', index=False, encoding='utf-8-sig')
