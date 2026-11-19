# M-104：120MA／200MA 平行、發散、收斂三種狀態下的訊號品質（觀察式，探索集）。定義見 registry_M.md M-104。
# 產出：lab/cache/M104/plot.json（作圖用彙總，不入庫）＋ summary。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals, sim

START, END = pd.Timestamp('2005-01-01'), pd.Timestamp('2015-12-31')
K, HX = 60, 120
NAMES = ['收斂', '平行', '發散', '非上升']
OUT = LAB / 'cache' / 'M104'


def states(g, dg, s120, s200, c):
    up = (s120 > 0) & (s200 > 0) & np.isfinite(dg)
    st = np.full(g.shape, 3, np.int8)
    st[up & (dg < -c)] = 0; st[up & (np.abs(dg) <= c)] = 1; st[up & (dg > c)] = 2
    return st


def q(a, ps=(10, 50, 90)):
    return [np.round(np.nanpercentile(a, p, axis=0), 5).tolist() for p in ps] if len(a) else [[], [], []]


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']
    U, pool, signal, Tr = dbg['U'], dbg['pool'], dbg['signal'], dbg['Tr']
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cfd = pd.DataFrame(CA).ffill(); Cf = Cfd.to_numpy()
    m120 = Cfd.rolling(120, min_periods=108).mean(); m200 = Cfd.rolling(200, min_periods=180).mean()
    g = (m120 / m200 - 1).to_numpy()
    lag = lambda A, k=20: np.vstack([np.full((k, A.shape[1]), np.nan), A[:-k]])
    dg = g - lag(g); s120 = (m120 / m120.shift(20) - 1).to_numpy(); s200 = (m200 / m200.shift(20) - 1).to_numpy()
    ST = states(g, dg, s120, s200, 0.01)
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    sid = P['meta']['stock_id'].astype(str); odd = (sid.str[-1].astype(int) % 2 == 1).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1

    # ---------- 1. 相關 ----------
    prk = signals.prk
    dT = (pd.DataFrame(Tr) - pd.DataFrame(Tr).shift(10)).to_numpy()
    PT, PD = prk(Tr, pool), prk(dT, pool)
    R20 = np.asarray(F['R20']); S60 = np.asarray(F['S60'])
    vars_ = dict(Δg=dg, g=g, T_ld百分位=PT, ΔT_ld百分位=PD, S60=S60, 漲幅20=R20)
    cms = []
    for t in range(t_lo, t_hi + 1, 21):
        m = pool[t] & odd
        if m.sum() < 15: continue
        cms.append(pd.DataFrame({k: v[t][m] for k, v in vars_.items()}).rank().corr().values)
    cm = pd.DataFrame(np.nanmean(np.array(cms), axis=0), index=list(vars_), columns=list(vars_))
    share = {NAMES[k]: float((ST[t_lo:t_hi + 1][pool[t_lo:t_hi + 1] & odd[None, :]] == k).mean()) for k in range(4)}

    # ---------- 2. V4.1 事件 ----------
    tt, jj = np.nonzero(signal)
    ok = (tt - K >= t_lo) & (tt + K <= t_hi) & odd[jj]; tt, jj = tt[ok], jj[ok]
    off = np.arange(-K, K + 1)
    M = CX[tt[:, None] + off[None, :], jj[:, None]] - CX[tt, jj][:, None]
    r20 = M[:, K + 20]; lo10, hi10 = np.percentile(r20, [10, 90])
    def evstats(mask):
        Mm = M[mask]; x = r20[mask]
        return dict(n=int(mask.sum()), mean=np.round(Mm.mean(0), 5).tolist() if len(Mm) else [], pct=q(Mm),
                    pre20=float(-Mm[:, K - 20].mean()) if len(Mm) else None, r20=float(x.mean()) if len(x) else None,
                    r60=float(Mm[:, K + 60].mean()) if len(Mm) else None, win20=float((x > 0).mean()) if len(x) else None,
                    left=float((x <= lo10).mean()) if len(x) else None, right=float((x >= hi10).mean()) if len(x) else None)
    est = ST[tt, jj]
    ev = {NAMES[k]: evstats(est == k) for k in range(4)}; ev['全部'] = evstats(np.ones(len(tt), bool))

    # ---------- 3. X1 交易 ----------
    Rp = sim.run_paths('fix', 100, sig='fix')
    kix = {k: j for j, k in enumerate(P['meta']['security_key'])}
    seen = {}
    for i in range(100):
        for (code, d0, d1, w, ret, hold, why, st, lk) in Rp['X'][i][1]:
            seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret, hold=hold, why=why))
    rows = []
    for v in seen.values():
        j = kix[v['code']]; t0 = int(dates.get_loc(v['d0'])); t1 = int(dates.get_loc(v['d1']))
        if not odd[j] or t0 < t_lo or t1 > t_hi: continue
        ent = OA[t0, j]
        if not np.isfinite(ent) or ent <= 0: continue
        hp = Cf[t0:t1 + 1, j] / ent - 1; pk = float(np.nanmax(hp))
        fwd = (Cf[t0:t0 + HX + 1, j] / ent - 1) if t0 + HX <= t_hi else None
        rows.append(dict(j=j, t0=t0, t1=t1, ret=v['ret'], hold=v['hold'], why=v['why'], mfe=pk, give=pk - v['ret'], fwd=fwd,
                         st=int(ST[t0 - 1, j]), st05=int(states(g[t0 - 1:t0, j:j+1], dg[t0 - 1:t0, j:j+1], s120[t0 - 1:t0, j:j+1], s200[t0 - 1:t0, j:j+1], 0.005)[0, 0]),
                         st20=int(states(g[t0 - 1:t0, j:j+1], dg[t0 - 1:t0, j:j+1], s120[t0 - 1:t0, j:j+1], s200[t0 - 1:t0, j:j+1], 0.02)[0, 0])))
    X = pd.DataFrame(rows)
    top, bot = X.ret >= X.ret.quantile(0.9), X.ret <= X.ret.quantile(0.1)
    gross = X.ret.clip(lower=0).sum()
    # X1 子帳戶三次最大回撤期間（2005–15，100 路徑平均權益）
    xr = pd.concat([Rp['X'][i][0] for i in range(100)], axis=1).mean(axis=1).loc[START:END]
    eq = (1 + xr).cumprod(); dd = eq / eq.cummax() - 1
    eps = []; used = pd.Series(False, index=dd.index)
    for _ in range(3):
        cand = dd[~used]
        if cand.empty or cand.min() >= 0: break
        tr_ = cand.idxmin(); pk_ = eq.loc[:tr_].idxmax()
        eps.append((pk_, tr_, float(dd.loc[tr_]))); used.loc[pk_:tr_] = True
        nxt = eq.loc[tr_:]; rec = nxt[nxt >= eq.loc[pk_]]
        used.loc[tr_:(rec.index[0] if len(rec) else dd.index[-1])] = True
    def ddcontrib(sub):
        tot = 0.0
        for pk_, tr_, _ in eps:
            a, b = int(dates.get_loc(pk_)), int(dates.get_loc(tr_))
            for _, r in sub.iterrows():
                s_, e_ = max(r.t0, a), min(r.t1, b)
                if s_ < e_: tot += Cf[e_, r.j] / Cf[s_, r.j] - 1
        return tot
    def xstats(sub):
        fw = np.array([f for f in sub.fwd if f is not None])
        win = np.array([f is not None and r > 0 for f, r in zip(sub.fwd, sub.ret)])[[f is not None for f in sub.fwd]] if len(sub) else np.array([], bool)
        return dict(n=int(len(sub)), mean=float(sub.ret.mean()), med=float(sub.ret.median()), win=float((sub.ret > 0).mean()),
                    top=float(top[sub.index].mean()), bot=float(bot[sub.index].mean()), mfe=float(sub.mfe.median()), give=float(sub.give.median()),
                    hold=float(sub.hold.mean()), share=float(sub.ret.clip(lower=0).sum() / gross), dd=ddcontrib(sub),
                    why=sub.why.value_counts().to_dict(), fan=q(fw), fan_win=q(fw[win]) if win.any() else [[], [], []],
                    fan_loss=q(fw[~win]) if (~win).any() else [[], [], []], nfan=int(len(fw)),
                    hist=np.histogram(sub.ret.clip(-0.4, 1.2), bins=np.arange(-0.4, 1.21, 0.08))[0].tolist())
    xs = {NAMES[k]: xstats(X[X.st == k]) for k in range(4)}; xs['全部'] = xstats(X)
    rob = {}
    for col, c in (('st05', '0.5pt'), ('st', '1pt'), ('st20', '2pt')):
        rob[c] = {NAMES[k]: dict(n=int((X[col] == k).sum()), mean=float(X.ret[X[col] == k].mean()) if (X[col] == k).any() else None,
                                 top=float(top[X[col] == k].mean()) if (X[col] == k).any() else None) for k in range(4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plot.json').write_text(json.dumps(dict(ev=ev, xs=xs, cm=cm.round(3).to_dict(), share=share, rob=rob, K=K, HX=HX,
                                                    eps=[(str(a.date()), str(b.date()), d) for a, b, d in eps], bins=np.arange(-0.4, 1.21, 0.08).round(2).tolist()),
                                               ensure_ascii=False, default=float), encoding='utf-8')

    L = ['# M-104 結果：120MA／200MA 三種狀態下的訊號品質（觀察式，探索集）\n',
         '強勢池探索集內狀態比例：' + '、'.join(f'{k} {v:.0%}' for k, v in share.items()) + '\n',
         '## 1. 相關（池內 Spearman，每 21 日一期平均）\n', '| | ' + ' | '.join(cm.columns) + ' |', '|---' * (len(cm.columns) + 1) + '|']
    for i, r in cm.iterrows(): L.append(f'| {i} | ' + ' | '.join(f'{v:+.2f}' for v in r.values) + ' |')
    L += ['\n## 2. V4.1 事件（後 20 日超額；左右尾以全部事件的 10／90 百分位為界）\n',
          '| 狀態 | 事件數 | 事件前 20 日 | 後 20 日 | 勝率 | 左尾率 | 右尾率 | 後 60 日 |', '|---|---|---|---|---|---|---|---|']
    for k, v in ev.items():
        if v['n']: L.append(f"| {k} | {v['n']} | {v['pre20']:+.1%} | {v['r20']:+.2%} | {v['win20']:.0%} | {v['left']:.0%} | {v['right']:.0%} | {v['r60']:+.2%} |")
    L += ['\n## 3. X1 交易（含停損的實際報酬）\n',
          '| 狀態 | 筆數 | 平均 | 中位數 | 勝率 | 落在前 10% | 落在後 10% | 最高浮盈中位數 | 回吐中位數 | 平均持有日 | 占總獲利 | 三次最大回撤期間損益 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, v in xs.items():
        if v['n']: L.append(f"| {k} | {v['n']} | {v['mean']:+.1%} | {v['med']:+.1%} | {v['win']:.0%} | {v['top']:.0%} | {v['bot']:.0%} | {v['mfe']:+.1%} | {v['give']:.1%} | {v['hold']:.0f} | {v['share']:.0%} | {v['dd']:+.2f} |")
    L.append('\n三次最大回撤（X1 子帳戶 100 路徑平均）：' + '；'.join(f'{a.date()}→{b.date()} {d:.1%}' for a, b, d in eps))
    L += ['\n穩健性（切點 0.5／1／2 個百分點；筆數、平均報酬、落在前 10%）：\n', '| 切點 | ' + ' | '.join(NAMES) + ' |', '|---' * 5 + '|']
    for c, d in rob.items():
        L.append(f'| {c} | ' + ' | '.join(f"{d[k]['n']}／{d[k]['mean']:+.1%}／{d[k]['top']:.0%}" if d[k]['n'] else '0' for k in NAMES) + ' |')
    return dict(summary='\n'.join(L), headline='觀察式作圖完成（見私人網頁）', n_variants=0)
