"""密集區突破：1) 頻率 2) 同股時機檢驗 3) 組合（每日進場、前 K 名內）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('port_timing.py').read().split("\nif __name__ == '__main__'")[0])
from dense import dense_signals
from factors import base
from evaluate import prepare
atr_ = base(P)['atr']
E2 = prepare(P)
ex = {h: E2['fwd'][h]['raw'].sub(E2['fwd'][h]['raw'].where(U).mean(1), axis=0).values for h in (20, 60)}
VAR = [(W, q, e) for W in (10, 20) for q in (1.0, 0.3) for e in ('dense', 'box')]
rows = []
for W, q, e in VAR:
    sig = dense_signals(P, atr_, U, W, q, e)
    sv = sig.values
    lab = f'W{W} {"密集上緣" if e == "dense" else "箱頂"} {"不限整理" if q == 1 else "整理≤30分位"}'
    # 頻率
    f50 = np.mean([(sv[t] & (RKall[t] <= 50)).sum() for t in range(d0, len(dates))]) * 252
    # 同股時機：前 100 名內的訊號日 vs 同股前後 60 日其他前 100 名日子
    T, J = np.where(sv & (RKall <= 100)); keep = T >= d0
    a = []
    for t, j in zip(T[keep], J[keep]):
        lo, hi = max(d0, t - 60), min(len(dates) - 61, t + 60)
        if hi <= lo: continue
        w = np.arange(lo, hi); w = w[(RKall[w, j] <= 100) & (np.abs(w - t) > 5) & ~sv[w, j]]
        if len(w) < 10: continue
        a.append([ex[20][t, j], np.nanmean(ex[20][w, j]), ex[60][t, j], np.nanmean(ex[60][w, j])])
    a = pd.DataFrame(a, columns=['e20', 'o20', 'e60', 'o60']).dropna(); d60 = a.e60 - a.o60
    r = dict(訊號=lab, 前50名每年次數=round(f50), 樣本=len(a), 訊號日60=round(a.e60.mean() * 100, 2), 其他日60=round(a.o60.mean() * 100, 2),
             差t=round(d60.mean() / d60.std() * np.sqrt(len(a) / 6), 1), 訊號日20=round(a.e20.mean() * 100, 2), 其他日20=round(a.o20.mean() * 100, 2))
    # 組合：每日進場
    for K in (30, 100):
        res = [run_evt(sig, K, off)[0] for off in range(0, 20, 4)]
        st = pd.DataFrame([stats(s) for s in res]).median()
        r[f'K{K} 年化'] = round(st.CAGR * 100, 1); r[f'K{K} 回撤'] = round(st.MDD * 100, 1); r[f'K{K} Sharpe'] = round(st.Sharpe, 2)
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/dense_test.csv', index=False, encoding='utf-8-sig')
