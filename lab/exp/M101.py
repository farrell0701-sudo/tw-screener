# M-101：觀察式訊號作圖（探索集：奇數代碼、2005–15）。定義見 registry_M.md M-101。
# 產出：lab/cache/M101/plot.json（作圖用彙總數據，不入庫）＋ summary（run.py 寫入 lab/results/M101/）。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals, sim

END = pd.Timestamp('2015-12-31'); START = pd.Timestamp('2005-01-01')
K = 60; HX = 120
OUT = LAB / 'cache' / 'M101'


def q(a, ps=(10, 50, 90)):
    return [np.round(np.nanpercentile(a, p, axis=0), 5).tolist() for p in ps]


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; daily = S['daily']
    U, pool, signal, sig0 = dbg['U'], dbg['pool'], dbg['signal'], dbg['sig0']
    deep = sig0 & ~signal                                      # 剔除前訊號中被深回撤剔掉的（sig0 & deep）
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0                       # 停牌日沿用最後價（報酬 0）
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nanmean(np.where(Ush, R, np.nan), axis=1)
    ew = np.nan_to_num(ew); XR = R - ew[:, None]                # 個股日超額
    CX = np.cumsum(XR, axis=0)                                  # 累積超額（對數近似用簡單加總，視窗內差分）
    sid = P['meta']['stock_id'].astype(str); odd = (sid.str[-1].astype(int) % 2 == 1).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1
    EV1, EV2 = np.asarray(F['EV1']), np.asarray(F['EV2']); MD = daily['bear_mode'].to_numpy(bool)

    # ---------- 圖 1：V4.1 事件研究 ----------
    def paths(mask):
        tt, jj = np.nonzero(mask)
        ok = (tt - K >= t_lo) & (tt + K <= t_hi) & odd[jj]
        tt, jj = tt[ok], jj[ok]
        off = np.arange(-K, K + 1)
        M = CX[tt[:, None] + off[None, :], jj[:, None]] - CX[tt, jj][:, None]
        return M, tt, jj
    ev = {}
    for name, m in [('全部', signal), ('布林收斂', signal & EV1), ('整理突破', signal & EV2),
                    ('多頭期', signal & ~MD[:, None]), ('空頭期', signal & MD[:, None]), ('被剔深回撤', deep)]:
        M, tt, jj = paths(m)
        ev[name] = dict(n=int(len(M)), mean=np.round(M.mean(0), 5).tolist(), pct=q(M),
                        win20=float((M[:, K + 20] > 0).mean()), r20=float(M[:, K + 20].mean()), r60=float(M[:, K + 60].mean()),
                        pre20=float(M[:, K - 20].mean() * -1))
    # ---------- 圖 2：X1 持有路徑 ----------
    Rr = sim.run_paths('fix', 100, sig='fix')
    seen = {}
    key_ix = {k: j for j, k in enumerate(P['meta']['security_key'])}
    for i in range(100):
        for (code, d0, d1, w, ret, hold, why, st, lk) in Rr['X'][i][1]:
            k = (code, d0)
            if k in seen: seen[k]['n'] += 1; continue
            seen[k] = dict(code=code, d0=d0, d1=d1, ret=ret, hold=hold, why=why, n=1)
    rows = []
    for v in seen.values():
        j = key_ix[v['code']]; t0 = int(dates.get_loc(v['d0'])); t1 = int(dates.get_loc(v['d1']))
        if not odd[j] or t0 < t_lo or t1 > t_hi: continue
        ent = OA[t0, j]
        if not np.isfinite(ent) or ent <= 0: continue
        hp = Cf[t0:t1 + 1, j] / ent - 1
        post = lambda h: float(CX[min(t1 + h, t_hi), j] - CX[t1, j]) if t1 + h <= t_hi else np.nan
        fwd = Cf[t0:t0 + HX + 1, j] / ent - 1 if t0 + HX <= t_hi else None
        rows.append(dict(ret=v['ret'], hold=v['hold'], why=v['why'], mfe=float(np.nanmax(hp)), mae=float(np.nanmin(hp)),
                         post20=post(20), post60=post(60), fwd=fwd, paths=v['n']))
    X = pd.DataFrame(rows)
    fw = np.array([r for r in X['fwd'] if r is not None])
    win = np.array([r is not None and x > 0 for r, x in zip(X['fwd'], X['ret'])])[[r is not None for r in X['fwd']]]
    reasons = X.groupby('why').agg(n=('ret', 'size'), ret=('ret', 'mean'), med=('ret', 'median'), hold=('hold', 'mean'),
                                   post20=('post20', 'mean'), post60=('post60', 'mean'), mfe=('mfe', 'mean')).sort_values('n', ascending=False)
    x1 = dict(n=int(len(X)), fan_all=q(fw), fan_win=q(fw[win]), fan_loss=q(fw[~win]), n_win=int(win.sum()), n_loss=int((~win).sum()),
              pts=[[round(a, 4), round(b, 4), round(c, 4), w] for a, b, c, w in zip(X.mfe, X.mae, X.ret, X.why)],
              reasons=reasons.reset_index().round(4).to_dict('records'),
              hold_hist=np.histogram(X.hold, bins=[0, 5, 10, 20, 40, 60, 90, 120, 180, 251])[0].tolist(),
              top10_share=float(X.ret.clip(lower=0).nlargest(max(1, len(X) // 10)).sum() / X.ret.clip(lower=0).sum()))
    # ---------- 圖 3：訊号密度與大盤 ----------
    rng = slice(t_lo, t_hi + 1)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill().to_numpy()
    s20 = pd.Series(signal.sum(1)).rolling(20, min_periods=1).sum().to_numpy()
    with np.errstate(all='ignore'):
        pew = np.nanmean(np.where(np.vstack([np.zeros((1, N), bool), pool[:-1]]), R, np.nan), axis=1)
    pex = np.nan_to_num(pew) - ew                                # 強勢池等權日超額
    fpool = pd.Series(pex).rolling(20).sum().shift(-20).to_numpy()   # 後續 20 日（t+1..t+20）
    fidx = pd.Series(ix).pct_change(20).shift(-20).to_numpy()
    tt = np.arange(T); ok = (tt >= t_lo) & (tt + 20 <= t_hi)
    def bins(x):
        b = pd.qcut(pd.Series(x[ok]), 5, labels=False, duplicates='drop')
        g = pd.DataFrame(dict(b=b.values, x=x[ok], fp=fpool[ok], fi=fidx[ok])).groupby('b')
        return g.agg(lo=('x', 'min'), hi=('x', 'max'), n=('x', 'size'), fp=('fp', 'mean'), fi=('fi', 'mean')).round(5).to_dict('records')
    wk = pd.Series(np.arange(T), index=dates).iloc[rng].resample('W-FRI').last().dropna().astype(int).to_numpy()
    dens = dict(date=[str(dates[t].date()) for t in wk], idx=np.round(ix[wk] / ix[t_lo] * 100, 2).tolist(),
                pool=daily['n_pool'].to_numpy()[wk].tolist(), candX=daily['n_candX'].to_numpy()[wk].tolist(),
                s20=s20[wk].astype(int).tolist(), bear=MD[wk].astype(int).tolist(),
                bins_sig=bins(s20), bins_cand=bins(daily['n_candX'].to_numpy().astype(float)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plot.json').write_text(json.dumps(dict(ev=ev, x1=x1, dens=dens, K=K, HX=HX), ensure_ascii=False, default=float), encoding='utf-8')

    L = ['# M-101 結果：訊號作圖（觀察式，探索集）\n', '圖在私人網頁；本摘要只列關鍵數字。期間 2005-01～2015-12，奇數代碼。\n',
         '## 圖 1：V4.1 事件（事件後累積超額，平均）\n', '| 組 | 事件數 | 前 20 日 | 後 20 日 | 後 20 日勝率 | 後 60 日 |', '|---|---|---|---|---|---|']
    for k, v in ev.items():
        L.append(f"| {k} | {v['n']} | {v['pre20']:+.1%} | {v['r20']:+.2%} | {v['win20']:.0%} | {v['r60']:+.2%} |")
    L += ['\n## 圖 2：X1 交易（不重複）\n', f"筆數 {x1['n']}；最終賺 {x1['n_win']}、賠 {x1['n_loss']}（有 120 日後續者）；前 10% 交易占總獲利 {x1['top10_share']:.0%}。\n",
          '| 出場原因 | 筆數 | 平均報酬 | 中位數 | 平均持有日 | 出場後 20 日超額 | 出場後 60 日超額 |', '|---|---|---|---|---|---|---|']
    for r in x1['reasons']:
        L.append(f"| {r['why']} | {r['n']} | {r['ret']:+.1%} | {r['med']:+.1%} | {r['hold']:.0f} | {r['post20']:+.2%} | {r['post60']:+.2%} |")
    L += ['\n## 圖 3：訊號密度分組（後續 20 日；日資料重疊，只作描述）\n', '| 分組 | 範圍 | 強勢池超額 | 指數報酬 |', '|---|---|---|---|']
    for nm, bb in [('V4.1 近 20 日訊號數', dens['bins_sig']), ('X1 候選數', dens['bins_cand'])]:
        for i, b in enumerate(bb):
            L.append(f"| {nm} Q{i + 1} | {b['lo']:.0f}–{b['hi']:.0f} | {b['fp']:+.2%} | {b['fi']:+.2%} |")
    return dict(summary='\n'.join(L), headline='觀察式作圖完成（見私人網頁）', n_variants=0)
