"""短線事件策略：強勢股出現事件 → 隔日開盤買，固定持有 H 日（可加 ATR 停損），10 個部位，依 T_ldf 排名先買"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
EV = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
strong = (cur.rank(axis=1, pct=True) >= 0.9) & ~ldn5
def short_eng(sig, label, H=20, k=None, n=10, top=None):
    """sig：DataFrame bool（當天收盤出現事件）；top：不看事件，改買 T_ldf 前 top 名（對照）"""
    S = (strong & sig.reindex_like(strong).fillna(False)) if sig is not None else (cur.where(~ldn5).rank(axis=1, ascending=False) <= top)
    RKs = cur.where(S & BUYdf).rank(axis=1, ascending=False).values
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan); T = []
    for t in range(d0, len(dates)):
        keep = set()
        for j in sell_next:
            if j not in pos: continue
            if lockdn[t, j]: keep.add(j); continue
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX); T.append(px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1)
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or lockup[t, j]: continue
                amt = min(val / n, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
        sell_next, buy_next = keep, []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        for j, p in pos.items():
            c = Cv[t, j]
            if j in sell_next: continue
            if t - p[3] + 1 >= H: sell_next.add(j); continue          # 持有滿 H 日，隔天開盤出場
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            if k and c < p[2] - k * Av[t, j]: sell_next.add(j)
        free = n - len(pos) + len(sell_next & set(pos))
        if free > 0:
            cand = np.where(~np.isnan(RKs[t]))[0]
            buy_next = [j for j in cand[np.argsort(RKs[t, cand])] if j not in pos][:free]
    s = pd.Series(eq, index=dates).dropna(); summ([s], label)
    T = np.array(T); out[-1]['交易數/年'] = round(len(T) / 10.7); out[-1]['勝率'] = f'{(T > 0).mean():.0%}'; out[-1]['每筆平均'] = f'{T.mean():.2%}'
    return s
anyev = EV['布林：收斂後首次突破上軌'] | EV['布林：收斂後帶寬張開 + 收上軌'] | EV['整理後突破 20 日高']
for H in (15, 25, 30, 40):
    short_eng(anyev, f'三種事件任一，持有 {H} 日', H=H)
for n_ in (5, 15):
    short_eng(anyev, f'三種事件任一，持有 20 日，{n_} 個部位', n=n_)
D = pd.DataFrame(out).drop(columns=['資料', '最低', '最差回撤'], errors='ignore'); pd.set_option('display.width', 250); print(D.to_string())
