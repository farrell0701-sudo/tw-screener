# V4.1 事件層共用：抽出全部 V4.1 最終訊號事件（全期、奇偶代碼），附結果變數與常用因子。M-156、M-157 使用。
# 結果變數同 M-155：Y＝C[t+20]÷O[t+1]−1−可交易池等權 t→t+20；MAE；STOP＝含停損版（空頭 5%／IAKF，次一有效日開盤出，扣 0.77%）。
# 因子：A1 收盤距 MA25、A6 T_ld 可交易池內百分位、A22 近 20 日 V4.1 訊號數（全部股票）。條件同 M-154：自身有效日 ≥ 100、t+1 有效。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals

H = 20; COST = 0.0077


def stat(d):
    y = d.Y.to_numpy()
    if len(y) == 0: return dict(n=0, mean=np.nan, med=np.nan, win=np.nan, succ=np.nan, fake=np.nan, lt=np.nan, rt=np.nan, mae=np.nan, stop=np.nan)
    return dict(n=len(y), mean=y.mean(), med=np.median(y), win=(y > 0).mean(), succ=(y >= .05).mean(), fake=(y <= -.05).mean(),
                lt=(y <= -.10).mean(), rt=(y >= .15).mean(), mae=np.nanmedian(d.MAE), stop=d.STOP.mean())


def removal(d, rm, per_col, periods):
    """被剔組（rm 為真）vs 保留組，回傳 (a)–(d) 判定與明細。"""
    kp = ~rm; a, k = stat(d[rm]), stat(d[kp])
    nlt = max(1, int((d.Y <= -.10).sum()))
    lt_share = int((rm & (d.Y <= -.10)).sum()) / nlt
    per = {p: (d[rm & (d[per_col] == p)].Y.mean(), d[kp & (d[per_col] == p)].Y.mean()) for p in periods}
    ok = dict(a=bool(a['mean'] < k['mean'] and all(x < y for x, y in per.values())), b=bool(a['rt'] <= k['rt']),
              c=bool(lt_share >= rm.mean()), d=bool(a['stop'] < k['stop']))
    return dict(rm=a, kp=k, frac=float(rm.mean()), lt_share=lt_share, per=per, ok=ok, all=all(ok.values()))


def events():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; daily = S['daily']
    U, signal, Tr = dbg['U'], dbg['signal'], dbg['Tr']
    MD = daily['bear_mode'].to_numpy(bool); OFF = daily['off'].to_numpy(bool)
    br20 = daily['n_signal'].rolling(20, min_periods=1).sum().to_numpy()
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    LA = np.asarray(F['LA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CEW = np.cumprod(1 + ew)
    sid = P['meta']['stock_id'].astype(str); par = (sid.str[-1].astype(int) % 2).to_numpy()
    t0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
    tt, jj = np.nonzero(signal); ok = (tt >= t0) & (tt + H <= T - 1); tt, jj = tt[ok], jj[ok]
    pr6 = {}
    for t in np.unique(tt):
        a = pd.Series(np.where(U[t], Tr[t], np.nan)).rank(pct=True).to_numpy()
        for j in jj[tt == t]: pr6[(t, j)] = a[j]
    rows = []
    for j in np.unique(jj):
        ii = np.flatnonzero(VALID[:, j]); pos = {t: k for k, t in enumerate(ii)}
        m25 = pd.Series(CA[ii, j]).rolling(25).mean().to_numpy()
        for t in tt[jj == j]:
            k = pos.get(t)
            if k is None or k < 100: continue
            ent = OA[t + 1, j]
            if not (VALID[t + 1, j] and np.isfinite(ent) and ent > 0): continue
            y = Cf[t + H, j] / ent - 1 - (CEW[t + H] / CEW[t] - 1)
            seg = LA[t + 1:t + H + 1, j]; mae = np.nanmin(seg) / ent - 1 if np.isfinite(seg).any() else np.nan
            ex = None
            for d in range(t + 1, t + H + 1):
                if not VALID[d, j]: continue
                if OFF[d] or ((MD[d] or MD[t]) and CA[d, j] < ent * 0.95):
                    nx = d + 1
                    while nx < T and not VALID[nx, j]: nx += 1
                    ex = OA[nx, j] if nx < T and np.isfinite(OA[nx, j]) else CA[d, j]
                    break
            if ex is None: ex = Cf[t + H, j]
            rows.append(dict(t=int(t), date=dates[t], year=int(dates[t].year), par=int(par[j]), Y=float(y), MAE=float(mae),
                             STOP=float(ex / ent - 1 - COST), A1=float(CA[t, j] / m25[k] - 1), A6=float(pr6[(t, j)]), A22=float(br20[t])))
    return pd.DataFrame(rows)
