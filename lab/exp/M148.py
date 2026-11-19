# M-148：M-145 的偶數代碼確認集（只測一次）。定義見 registry_M.md M-148。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M144 import boot

NP = 100; ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def main():
    rng = np.random.default_rng(20261122)
    P = load_panel(); F = load_feat('AF')
    signals.build('fix', mode='fix', P=P, F=F); signals.build('vbw21', mode='fix', P=P, F=F, v_bearw=21)
    odd = {k: int(str(c)[-1]) % 2 == 0 for k, c in zip(P['meta']['security_key'], P['meta']['stock_id'].astype(str))}
    def dedup(R):
        seen = {}
        for i in range(NP):
            for (code, d0, d1, w, ret, *_r) in R['V'][i][1]: seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret))
        X = pd.DataFrame(seen.values()); X['odd'] = X.code.map(odd); X['m'] = X.d0.astype(str).str[:7]; return X.set_index(['code', 'd0'], drop=False)
    B = dedup(sim.run_paths('fix', NP, sig='fix')); N_ = dedup(sim.run_paths('fix', NP, sig='vbw21'))
    L = ['# M-148 結果：V4.1 空頭模式改用週 EMA21 下彎——確認集（**偶數代碼**，只測一次）\n']
    verdict = None
    for era, lo, hi in ERAS:
        f = lambda X: X[X.odd & (X.d0 >= lo) & (X.d1 <= hi)]
        b, n = f(B), f(N_)
        ext = n.loc[n.index.difference(b.index)]; rem = b.loc[b.index.difference(n.index)]; com = b.index.intersection(n.index)
        q10, q90 = b.ret.quantile([0.1, 0.9]); p = boot(ext.ret.values, ext.m.values, rng) if len(ext) > 5 else np.nan
        dcom = (n.loc[com].ret - b.loc[com].ret)
        crit = [ext.ret.mean() > 0 and p >= 0.90, (ext.ret <= q10).mean() <= 0.15 and (ext.ret >= q90).mean() >= 0.10, rem.ret.mean() < b.ret.mean(), len(ext) >= 30, dcom.mean() >= 0]
        L += [f'## {era}｜偶數代碼（基準 V4.1 交易 {len(b)} 筆，平均 {b.ret.mean():+.2%}）\n', '| 項目 | 數值 | 判準 |', '|---|---|---|',
              f"| (a) 多出來的交易平均（P） | {ext.ret.mean():+.2%}（{p:.3f}），{len(ext)} 筆 | {'✓' if crit[0] else '✗'} |",
              f"| (b) 左尾率／右尾率 | {(ext.ret <= q10).mean():.0%}／{(ext.ret >= q90).mean():.0%} | {'✓' if crit[1] else '✗'} |",
              f"| (c) 少掉的交易平均 vs 基準全部 | {rem.ret.mean():+.2%}（{len(rem)} 筆） vs {b.ret.mean():+.2%} | {'✓' if crit[2] else '✗'} |",
              f"| (d) 多出來 ≥ 30 筆 | {len(ext)} | {'✓' if crit[3] else '✗'} |",
              f"| (e) 共同進場 新 − 舊 | {dcom.mean():+.2%}（{len(com)} 筆；有差異 {int((dcom.abs() > 1e-9).sum())} 筆） | {'✓' if crit[4] else '✗'} |", '']
        if era == '2005–15': verdict = all(crit)
    L.append(f"**確認集（偶數 2005–15）：{'通過，可登記 L 系列' if verdict else '不通過，結案'}**")
    return dict(summary='\n'.join(L), headline=f"V4.1 週 EMA 空頭模式確認集{'通過' if verdict else '不通過'}", n_variants=0)
