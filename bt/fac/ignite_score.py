"""短線事件策略：強勢股出現事件 → 隔日開盤買，固定持有 H 日（可加 ATR 停損），10 個部位，依 T_ldf 排名先買"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
EV = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
strong = (cur.rank(axis=1, pct=True) >= 0.9) & ~ldn5
def short_eng(sig, label, H=20, k=None, n=10, top=None, prio=None, stats=None):
    """sig：DataFrame bool（當天收盤出現事件）；top：不看事件，改買 T_ldf 前 top 名（對照）"""
    S = (strong & sig.reindex_like(strong).fillna(False)) if sig is not None else (cur.where(~ldn5).rank(axis=1, ascending=False) <= top)
    base_sc = cur if prio is None else prio
    RKs = base_sc.where(S & BUYdf).rank(axis=1, ascending=False, method='first').values
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
            cand2 = [j for j in cand[np.argsort(RKs[t, cand])] if j not in pos]
            if stats is not None and len(cand2): stats.append((len(cand2), free))
            buy_next = cand2[:free]
    s = pd.Series(eq, index=dates).dropna(); summ([s], label)
    T = np.array(T); out[-1]['交易數/年'] = round(len(T) / 10.7); out[-1]['勝率'] = f'{(T > 0).mean():.0%}'; out[-1]['每筆平均'] = f'{T.mean():.2%}'
    return s
anyev = EV['布林：收斂後首次突破上軌'] | EV['布林：收斂後帶寬張開 + 收上軌'] | EV['整理後突破 20 日高']
M = pd.read_pickle('/home/claude/bt/fac/ignite_mod.pkl')
bl, sl, vd = M['base_len'], M['sq_len'], M['vdry']
score = ((bl >= 10) & (bl < 40)).astype(int) + ((sl >= 3) & (sl < 15)).astype(int) + (vd < 0.9).astype(int) - (vd > 1.5).astype(int)
sc_ev = score.where(anyev & strong)
print('事件的分數分布：', sc_ev.stack().value_counts(normalize=True).sort_index().round(3).to_dict())
tp = cur.rank(axis=1, pct=True)
def ladder(sig, lab, prio=None, st=None):
    eqs = [short_eng(sig, f'{lab} H{H}', H=H, prio=prio, stats=st if H == 20 else None) for H in (15, 20, 25)]
    r = sum(e.pct_change() for e in eqs) / 3; r = r.dropna(); e = (1 + r).cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    tr = sum(o['交易數/年'] for o in out[-3:]) / 3
    print(f'{lab}：年化 {e.iloc[-1] ** (1 / y) - 1:.1%}，回撤 {(e / e.cummax() - 1).min():.1%}，Sharpe {r.mean() / r.std() * np.sqrt(252):.2f}，每年約 {tr:.0f} 筆', flush=True)
st = []
ladder(anyev, '基準：依 T_ldf 排序', st=st)
st = np.array(st); short = st[:, 0] > st[:, 1]
print(f'  有候選可買的日子 {len(st)} 天；其中候選多於空位 {short.mean():.0%}；這些天平均候選 {st[short, 0].mean():.1f} 檔、空位 {st[short, 1].mean():.1f} 個')
ladder(anyev, '想法1：同天先比分數，同分再比 T_ldf', prio=score + tp * 0.99)
ladder(anyev, '想法1b：T_ldf 百分位＋0.1×分數', prio=tp + 0.1 * score)
ladder(anyev, '想法1 反向對照：分數低的先買', prio=-score + tp * 0.99)
for th in (1, 2):
    ladder(anyev & (score >= th), f'想法2：分數 ≥{th} 才進場')
