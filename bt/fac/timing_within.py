"""同一檔股票的進場時機比較：事件日進場 vs 同一檔在前後 60 個交易日內、同樣排在前 K 名的其他日子進場"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('port_timing.py').read().split("\nif __name__ == '__main__'")[0])
from evaluate import prepare
E2 = prepare(P)
ex = {h: E2['fwd'][h]['raw'].sub(E2['fwd'][h]['raw'].where(U).mean(1), axis=0).values for h in (20, 60)}
cols = {c: i for i, c in enumerate(C.columns)}; di = {d: i for i, d in enumerate(dates)}
for name, kind in (('A出現', 'A出現即進'), ('上緣突破', '上緣突破進')):
    e = EV[EV.kind == kind]
    for K in (100, 300):
        rows = []
        for d, c in zip(e.date, e.code):
            t, j = di[d], cols[c]
            if not RKall[t, j] <= K: continue
            lo, hi = max(0, t - 60), min(len(dates) - 61, t + 60)
            if hi <= lo: continue
            w = np.arange(lo, hi); w = w[(RKall[w, j] <= K) & (np.abs(w - t) > 5)]
            if len(w) < 10: continue
            rows.append([ex[20][t, j], np.nanmean(ex[20][w, j]), ex[60][t, j], np.nanmean(ex[60][w, j])])
        a = pd.DataFrame(rows, columns=['ev20', 'oth20', 'ev60', 'oth60']).dropna()
        d20, d60 = (a.ev20 - a.oth20), (a.ev60 - a.oth60)
        print(f'{name} 前{K}名 n={len(a)}：事件日 20日 {a.ev20.mean()*100:+.2f}% vs 其他日 {a.oth20.mean()*100:+.2f}% | 60日 {a.ev60.mean()*100:+.2f}% vs {a.oth60.mean()*100:+.2f}%  差 t≈{d60.mean()/d60.std()*np.sqrt(len(a)/3):.1f}', flush=True)
