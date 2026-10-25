"""排除金融／營建、或在候選池內優先買高波動股（乾淨資料、每日補位引擎）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/rerun_clean.py').read().split("if __name__ == '__main__':")[0])
atrp = (atr / C).where(U)


def daily2(pool_sc, prio, label, K=30, n=10, kb=4, kr=3, offs=range(20), log_ind=False):
    """候選池 = pool_sc 前 K 名；池內依 prio 由高到低買"""
    RKp = pool_sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = pool_sc.rank(axis=1, ascending=False).values
    PR = prio.values
    res, inds = [], []
    for off in offs:
        reb = set(range(d0 + off, len(dates) - 2, REB))
        cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan)
        for t in range(d0, len(dates)):
            for j in sell_next:
                if j in pos:
                    px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                    cash += pos.pop(j)[0] * px * (1 - FEE - TAX)
            if buy_next:
                val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
                for j in buy_next:
                    if j in pos or len(pos) >= n: continue
                    px = Ov[t, j]
                    if np.isnan(px): continue
                    amt = min(val / n, cash)
                    if amt <= 0: break
                    pos[j] = [amt * (1 - FEE) / px, px, px]; cash -= amt
                    if log_ind and off == 0: inds.append((dates[t].year, indv.iloc[j] if isinstance(indv.iloc[j], str) else '無資料'))
            sell_next, buy_next = set(), []
            eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
            k = kb if BULL[t] else kr
            for j, p in pos.items():
                c = Cv[t, j]
                if np.isnan(c): continue
                p[2] = max(p[2], c)
                if c < p[2] - k * Av[t, j]: sell_next.add(j)
            if t in reb: sell_next |= {j for j in pos if not (RKall[t, j] <= 2 * K)}
            free = n - len(pos) + len(sell_next & set(pos))
            if free > 0:
                cand = np.where(RKp[t] <= K)[0]
                pr = np.nan_to_num(PR[t, cand], nan=-1e9)
                buy_next = [j for j in cand[np.argsort(-pr)] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna())
    summ(res, label)
    return inds


if __name__ == '__main__':
    fin = indv.isin(['金融保險業', '建材營造業']).reindex(C.columns).fillna(False).values
    FINdf = pd.DataFrame(np.repeat(fin[None, :], len(dates), 0), index=dates, columns=C.columns)
    inds = daily2(cur, cur, '每日補位｜現行（池內依 T_ldf）', log_ind=True)
    I = pd.DataFrame(inds, columns=['y', 'ind'])
    share = I.assign(fc=I.ind.isin(['金融保險業', '建材營造業'])).groupby('y').fc.mean()
    print('買進筆數中金融＋營建占比（起始日 0）：', ' '.join(f'{k}:{v:.0%}' for k, v in share.items()), flush=True)
    daily2(cur.where(~FINdf), cur, '排除金融＋營建')
    daily2(cur.where(~FINdf & 0 == 0), cur, '（對照）同上不排除')
    daily2(cur, atrp, '池內優先買高波動（ATR% 高者先）')
    daily2(cur, -atrp, '池內優先買低波動（對照）')
    mid_sc = ind_filter((((e12 - e200) / C) / np.sqrt(atr / C)).where(U))
    daily2(mid_sc, mid_sc, '分母改 √ATR%（介於 ATR 與百分比之間）')
    pct_sc = ind_filter(((e12 - e200) / C).where(U))
    daily2(pct_sc, pct_sc, '分母改股價（百分比距離）')
    for K in (15, 50):
        daily2(cur, atrp, f'池內優先高波動，K={K}', K=K)
    pd.DataFrame(out).to_csv('/home/claude/bt/fac/fin_const.csv', index=False, encoding='utf-8-sig')
