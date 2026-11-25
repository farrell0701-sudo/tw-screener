# M-201：強勢股集體崩跌的前兆。定義見 registry_M.md M-201。
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M165

END = pd.Timestamp('2024-11-13')
PARITY = int(os.environ.get('M201_PARITY', '1'))
ERAS = ('2005–15', '2016–24/11')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']
    par = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cdf = pd.DataFrame(CA).ffill(limit=5); C = Cdf.to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    R = np.where(VALID & np.isfinite(R), np.clip(R, -0.5, 1.0), np.nan)
    samp = sorted(t for _, (t, _) in M165.samp_prev(dates).items())
    mS = np.zeros((T, N), bool); mU = np.zeros((T, N), bool)
    for a, b in zip(samp, samp[1:] + [T - 1]):
        mS[a + 1:b + 1] = pool[a] & U[a] & par; mU[a + 1:b + 1] = U[a] & par
    with np.errstate(all='ignore'):
        rS = np.nan_to_num(np.nanmean(np.where(mS, R, np.nan), axis=1)); rU = np.nan_to_num(np.nanmean(np.where(mU, R, np.nan), axis=1))
    lS, lU = np.cumsum(np.log1p(rS)), np.cumsum(np.log1p(rU))
    def rel(t0, t1): return np.expm1(lS[t1] - lS[t0]) - np.expm1(lU[t1] - lU[t0])
    def uret(t0, t1): return np.expm1(lU[t1] - lU[t0])
    ATR = np.asarray(F['ATR'], np.float64)
    EMA50 = Cdf.ewm(span=50, adjust=False).mean().to_numpy()
    PU = pool & U & par
    tend = int(dates.searchsorted(END, side='right')) - 1; s0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
    rows = []
    for t in range(max(s0, 260), tend - 60, 5):
        m = PU[t]
        if m.sum() < 10: continue
        r20 = C[t, m] / C[t - 20, m] - 1
        rows.append(dict(t=t, era=ERAS[0] if dates[t] <= pd.Timestamp('2015-12-31') else ERAS[1],
                         y20=rel(t, t + 20), y60=rel(t, t + 60),
                         動能池20日=rel(t - 20, t), 動能池60日=rel(t - 60, t), 分散度=np.nanstd(r20), 汰換率=1 - PU[t - 20, m].mean(),
                         擁擠度EXT50=np.nanmean((C[t, m] - EMA50[t, m]) / ATR[t, m]), 大盤波動=np.std(rU[t - 59:t + 1]) * np.sqrt(252),
                         恐慌反彈=(uret(t - 250, t) < 0) and (uret(t - 20, t) > 0.05), 空頭狀態=uret(t - 250, t) < 0))
    D = pd.DataFrame(rows)
    q10 = D.y20.quantile(0.10); D['crash'] = D.y20 <= q10
    cont = ['動能池20日', '動能池60日', '分散度', '汰換率', '擁擠度EXT50', '大盤波動']; flg = ['恐慌反彈', '空頭狀態']
    def st(x): return f"{x.y20.mean():+.1%}｜{x.crash.mean():.0%}｜{x.y60.mean():+.1%}（{len(x)}）"
    out = [f"# M-201 結果：強勢股集體崩跌的前兆（{'奇' if PARITY else '偶'}數代碼；每 5 日取樣；到 2024-11-13）\n",
           f"結果 = 未來 20 日「強勢池 − U」。崩跌 = 最差 10%（≤ {q10:+.1%}）。格式：20 日平均｜崩跌率｜60 日平均（n）。\n",
           f"全體：{st(D[D.era == ERAS[0]])} ｜ {st(D[D.era == ERAS[1]])}\n",
           '| 條件 | 2005–15 旗標｜其他 | 2016–24/11 旗標｜其他 | 判定 |', '|---|---|---|---|']
    conds = []
    for k in cont:
        hi, lo = D[k].quantile(0.8), D[k].quantile(0.2)
        conds += [(f'{k} 最高 20%（≥ {hi:.3g}）', D[k] >= hi), (f'{k} 最低 20%（≤ {lo:.3g}）', D[k] <= lo)]
    conds += [(k, D[k].astype(bool)) for k in flg]
    noms = []
    for nm, f in conds:
        ok, cells = True, []
        for e in ERAS:
            a, b = D[f & (D.era == e)], D[~f & (D.era == e)]
            cells.append(f"{st(a)}<br>{st(b)}")
            ok &= len(a) >= 30 and a.y20.mean() <= b.y20.mean() - 0.01 and a.crash.mean() >= 1.5 * max(b.crash.mean(), 1e-9)
        if ok: noms.append(nm)
        out.append(f"| {nm} | {cells[0]} | {cells[1]} | {'**提名**' if ok else ''} |")
    big = D.nsmallest(15, 'y20')
    out += ['', '## 最差 15 個取樣日（看是否集中在少數事件）\n', '| 日期 | 未來 20 日 S−U | 恐慌反彈 | 空頭 | 分散度 | 汰換率 | 動能池 20 日 |', '|---|---|---|---|---|---|---|']
    out += [f"| {dates[int(r.t)]:%Y-%m-%d} | {r.y20:+.1%} | {'是' if r.恐慌反彈 else ''} | {'是' if r.空頭狀態 else ''} | {r.分散度:.3f} | {r.汰換率:.0%} | {r.動能池20日:+.1%} |" for r in big.itertuples()]
    out += ['', '## 提名\n'] + ([f'- {n}' for n in noms] or ['- 無'])
    thr = {k: (float(D[k].quantile(0.8)), float(D[k].quantile(0.2))) for k in cont}
    if PARITY: (LAB / 'results' / 'M201_thr.json').write_text(pd.Series(thr).to_json(force_ascii=False))
    return dict(summary='\n'.join(out), headline=('提名：' + '、'.join(noms)) if noms else '無提名', n_variants=0 if PARITY else len(noms))
