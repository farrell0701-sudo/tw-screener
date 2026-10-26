"""短線事件策略的多路徑引擎：起始日錯開（offs），每條路徑跑分批 15/20/25 日出場，取中位數
sig：DataFrame bool（當天收盤出現訊號）；prio：排序分數（越大越先買），預設 T_ldf"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
strong = (cur.rank(axis=1, pct=True) >= 0.9) & ~ldn5
EVd = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
anyev = (EVd['布林：收斂後首次突破上軌'] | EVd['布林：收斂後帶寬張開 + 收上軌'] | EVd['整理後突破 20 日高']).reindex_like(C).fillna(False)
def cand_lists(sig, prio=None, pool=None, drop=0.0, seed=0):
    pool = strong if pool is None else pool
    S = (pool & sig.reindex_like(C).fillna(False) & BUYdf).values
    if drop: S = S & (np.random.default_rng(seed).random(S.shape) >= drop)
    pr = (cur if prio is None else prio).reindex_like(C).values
    out = []
    for t in range(len(dates)):
        j = np.where(S[t] & ~np.isnan(pr[t]))[0]
        out.append(j[np.argsort(-pr[t, j], kind='stable')])
    return out
def run1(CL, H, start, n=10, W=None):
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan); rets = []
    for t in range(start, len(dates)):
        keep = set()
        for j in sell_next:
            if j not in pos: continue
            if lockdn[t, j]: keep.add(j); continue
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX); rets.append(px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1)
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or lockup[t, j]: continue
                w = 1.0 if W is None else W[t - 1, j]
                amt = min(val / n * w, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, t]; cash -= amt
        sell_next, buy_next = keep, []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        for j, p in pos.items():
            if j not in sell_next and t - p[2] + 1 >= H: sell_next.add(j)
        free = n - len(pos) + len(sell_next & set(pos))
        if free > 0:
            buy_next = [j for j in CL[t] if j not in pos][:free]
    return pd.Series(eq, index=dates).dropna(), rets
def stat(r):
    r = r.dropna(); e = (1 + r).cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    yr = e.resample('YE').last().pct_change(); yr.iloc[0] = e.resample('YE').last().iloc[0] - 1
    return dict(年化=e.iloc[-1] ** (1 / y) - 1, 回撤=(e / e.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252), 最差年=yr.min())
RES = []
def sleeve(sig, label, prio=None, pool=None, ladder=(15, 20, 25), npath=20, drop=0.1, n=10, W=None, start_date=None, keep=False):
    offs = range(npath)
    s0 = d0 if start_date is None else dates.get_indexer([pd.Timestamp(start_date)], method='bfill')[0]
    st, ntr, paths = [], [], []
    for o in offs:
        CL = cand_lists(sig, prio, pool, drop, seed=o); rs, nn = [], 0
        for H in ladder:
            e, rr = run1(CL, H, s0 + o % 5, n, None if W is None else W.reindex_like(C).values); rs.append(e.pct_change()); nn += len(rr)
        r = sum(rs) / len(rs); st.append(stat(r)); ntr.append(nn / len(ladder)); paths.append(r)
    D = pd.DataFrame(st); m = D.median(); y = (dates[-1] - dates[s0]).days / 365.25
    o = dict(版本=label, 年化=f'{m.年化:.1%}', 年化範圍=f'{D.年化.min():.1%}～{D.年化.max():.1%}', 回撤=f'{m.回撤:.1%}', 最差回撤=f'{D.回撤.min():.1%}',
             Sharpe=round(m.Sharpe, 2), 最差年=f'{m.最差年:.1%}', 每年交易=round(np.median(ntr) / y))
    RES.append(o); print(o, flush=True)
    return paths if keep else None
if __name__ == '__main__':
    import time; t0 = time.time()
    sleeve(anyev, '基準：三種收斂突破事件（多路徑）')
    print(f'{time.time() - t0:.0f} 秒')
