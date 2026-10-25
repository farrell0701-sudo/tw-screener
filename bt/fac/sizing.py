"""部位管理：急漲只買半個部位／等風險部位（基準＝真實執行＋近5日一字跌停不買）
每檔目標金額 = 淨值/10 × 乘數；乘數 < 1 時剩下的錢留現金（不多買一檔）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\ndef engine")[0])
def engine_w(sc, label, W=None, n=10, K=30, kb=4, kr=3, offs=range(20)):
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    Wv = None if W is None else W.reindex_like(sc).values
    res, expo = [], []
    for off in offs:
        reb = set(range(d0 + off, len(dates) - 2, REB))
        cash = 1.0; pos = {}; sell_next, buy_next = {}, []; eq = np.full(len(dates), np.nan); ex = []
        for t in range(d0, len(dates)):
            keep = {}
            for j, why in sell_next.items():
                if j not in pos: continue
                if lockdn[t, j]: keep[j] = why; continue
                px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
            if buy_next:
                val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
                for j in buy_next:
                    if j in pos or len(pos) >= n: continue
                    px = Ov[t, j]
                    if np.isnan(px) or lockup[t, j]: continue
                    w = 1.0 if Wv is None else Wv[t - 1, j]
                    if np.isnan(w): w = 1.0
                    amt = min(val / n * w, cash)
                    if amt <= 0: break
                    pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
            sell_next, buy_next = dict(keep), []
            eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items()); ex.append(1 - cash / eq[t])
            k = kb if BULL[t] else kr
            for j, p in pos.items():
                c = Cv[t, j]
                if np.isnan(c) or j in sell_next: continue
                p[2] = max(p[2], c)
                if c < p[2] - k * Av[t, j]: sell_next[j] = 'ATR'
                elif c < ML[t, j]: sell_next[j] = 'L'
            if t in reb:
                for j in pos:
                    if j not in sell_next and not (RKall[t, j] <= 2 * K): sell_next[j] = 'R'
            free = n - len(pos) + len(set(sell_next) & set(pos))
            if free > 0:
                cand = np.where(RK[t] <= K)[0]
                buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna()); expo.append(np.mean(ex))
    summ(res, label); out[-1]["平均持股比重%"] = round(np.median(expo) * 100); return res
if __name__ == '__main__':
    ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
    sc = cur.where(~ldn5)
    strong = cur.rank(axis=1, pct=True) >= 0.9
    # 急漲定義 1：之字波段上漲速度 ≥ 2%/天
    Fz = pd.read_pickle('/home/claude/bt/fac/swing_shape_1.5_lo.pkl')
    fast1 = (Fz['上漲速度'] >= np.log(1.02))
    # 急漲定義 2：log(120日高/低)/低到高天數，在強勢股中最快 25%
    HH = Hd.rolling(120, min_periods=100).max(); LL = Ld.rolling(120, min_periods=100).min()
    iH = Hd.rolling(120, min_periods=100).apply(np.argmax, raw=True); iL = Ld.rolling(120, min_periods=100).apply(np.argmin, raw=True)
    nd = iH - iL; sp = (np.log(HH / LL) / nd).where(nd > 0)
    fast2 = sp.where(strong).rank(axis=1, pct=True) > 0.75
    atrp = pd.DataFrame(Av, index=C.index, columns=codes) / C
    ref = atrp.where(strong).median(1)                         # 當天強勢股 ATR% 中位數 = 標準部位
    eqr = (1 / atrp).mul(ref, axis=0)
    engine_w(sc, '基準：等權')
    engine_w(sc, '急漲（波段速度≥2%/天）半個部位', fast1.astype(float).replace({1.0: 0.5, 0.0: 1.0}))
    engine_w(sc, '急漲（強勢股速度前25%）半個部位', fast2.astype(float).replace({1.0: 0.5, 0.0: 1.0}))
    engine_w(sc, '等風險：只縮不放（上限 1）', eqr.clip(upper=1.0))
    engine_w(sc, '等風險：上限 1.5 倍', eqr.clip(upper=1.5))
    engine_w(sc, '等風險：上限 2 倍', eqr.clip(upper=2.0))
    print(pd.DataFrame(out).drop(columns='資料').to_string())
