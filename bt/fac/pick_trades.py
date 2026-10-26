import sys; sys.path.insert(0, '/home/claude/bt/fac'); sys.path.insert(0, '/home/claude/bt')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
from load import load_all
NM = load_all()[0].set_index('code')['name']
EV = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
e1, e2, e3 = (EV[k].reindex_like(C).fillna(False) for k in ('布林：收斂後首次突破上軌', '布林：收斂後帶寬張開 + 收上軌', '整理後突破 20 日高'))
anyev = e1 | e2 | e3
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
strong = (cur.rank(axis=1, pct=True) >= 0.9) & ~ldn5
S = strong & anyev
RKs = cur.where(S & BUYdf).rank(axis=1, ascending=False, method='first').values
H, n = 20, 10
cash = 1.0; pos = {}; sell_next, buy_next = set(), []; T = []
E1, E2, E3 = e1.values, e2.values, e3.values
for t in range(d0, len(dates)):
    keep = set()
    for j in sell_next:
        if j not in pos: continue
        if lockdn[t, j]: keep.add(j); continue
        px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
        p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
        T.append(dict(j=j, 訊號日=dates[p[3] - 1].date(), 進場日=dates[p[3]].date(), 出場日=dates[t].date(), 持有=t - p[3],
                      進場價=round(p[1], 2), 出場價=round(px, 2), 報酬=px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1, 事件=p[4]))
    if buy_next:
        val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
        for j in buy_next:
            if j in pos or len(pos) >= n: continue
            px = Ov[t, j]
            if np.isnan(px) or lockup[t, j]: continue
            amt = min(val / n, cash)
            if amt <= 0: break
            evn = '＋'.join(k for k, a in (('①布林首次突破上軌', E1), ('②布林帶寬張開', E2), ('③整理後突破20日高', E3)) if a[t - 1, j])
            pos[j] = [amt * (1 - FEE) / px, px, px, t, evn]; cash -= amt
    sell_next, buy_next = keep, []
    for j, p in pos.items():
        if j not in sell_next and t - p[3] + 1 >= H: sell_next.add(j)
    free = n - len(pos) + len(sell_next & set(pos))
    if free > 0:
        cand = np.where(~np.isnan(RKs[t]))[0]
        buy_next = [j for j in cand[np.argsort(RKs[t, cand])] if j not in pos][:free]
T = pd.DataFrame(T); T['代號'] = codes[T.j]; T['名稱'] = [NM.get(c, '') for c in T.代號]
T.to_csv('/home/claude/bt/fac/short_trades_H20.csv', index=False, encoding='utf-8-sig')
print(f'交易 {len(T)} 筆：獲利 {(T.報酬 > 0).sum()}、虧損 {(T.報酬 <= 0).sum()}；平均 {T.報酬.mean():.2%}')
out_rows = []
for lab, X in (('虧損', T[T.報酬 <= 0].sort_values('報酬')), ('獲利', T[T.報酬 > 0].sort_values('報酬'))):
    k = len(X)
    for i in range(5):
        seg = X.iloc[int(k * i / 5):int(k * (i + 1) / 5)]
        r = seg.iloc[len(seg) // 2]
        rng_ = f'{seg.報酬.min():+.1%}～{seg.報酬.max():+.1%}'
        name = (f'最深 20%' if i == 0 else f'{i*20}～{(i+1)*20}%') if lab == '虧損' else f'{i*20}～{(i+1)*20}%' + ('（最大 20%）' if i == 4 else '')
        out_rows.append(dict(群組=lab, 百分位=name, 該組報酬範圍=rng_, 代號=r.代號, 名稱=r.名稱, 訊號日=r.訊號日, 進場日=r.進場日, 出場日=r.出場日,
                             進場價=r.進場價, 出場價=r.出場價, 報酬=f'{r.報酬:+.1%}', 事件=r.事件))
D = pd.DataFrame(out_rows); pd.set_option('display.width', 300); pd.set_option('display.max_columns', 20)
print(D.to_string(index=False))
