"""對照：同一個「每日進場」引擎，但不用密集區訊號（空位一出現就買池內排名最高的／隨機時點）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('port_timing.py').read().split("\nif __name__ == '__main__'")[0])
from dense import dense_signals
from factors import base
atr_ = base(P)['atr']
allT = U.copy()
rows = []
def go(sig, lab, K, offs=range(20)):
    res = [run_evt(sig, K, off)[0] for off in offs]
    st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s in res]); m = st.median()
    r = dict(版本=lab, K=K, 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1), 最差回撤=round(st.MDD.min() * 100, 1),
             Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1))
    rows.append(r); print(r, flush=True)
for K in (30,):
    go(allT, '無訊號：有空位就買池內最強', K)
    rng = np.random.default_rng(0)
    for p in (0.05, 0.03):
        rnd = pd.DataFrame(rng.random(U.shape) < p, index=dates, columns=C.columns) & U
        go(rnd, f'隨機時點（每天 {int(p*100)}% 機率）', K)
    go(dense_signals(P, atr_, U, 20, 1.0, 'dense'), 'W20 密集上緣', K)
    go(dense_signals(P, atr_, U, 20, 0.3, 'dense'), 'W20 密集上緣＋整理', K)
    go(dense_signals(P, atr_, U, 10, 0.3, 'dense'), 'W10 密集上緣＋整理', K)
    go(dense_signals(P, atr_, U, 20, 1.0, 'box'), 'W20 箱頂', K)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/dense_ctrl.csv', index=False, encoding='utf-8-sig')
