# M-103：X1「浮盈後收緊停損」交易層反事實試算（探索集）。定義與門檻見 registry_M.md M-103。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from sim import FEE, TAX, SLIP, _sellable, _sellpx

START, END = pd.Timestamp('2005-01-01'), pd.Timestamp('2015-12-31')
XS, YS = [0.08, 0.10, 0.15, 0.20], [0.05, 0.07, 0.10]
MAIN = (0.10, 0.07)


def replay(c, j, t0, t1, d, X=None, Y=None, cap=250):
    """重播一筆交易：t0 開盤買、依規則判斷，回傳（出場日, 報酬）。不晚於 t1；到 t1 仍未觸發則回傳 None（沿用原結果）。"""
    px = c['Ov'][t0, j] * (1 + SLIP); hi = px; sell_on = None
    for t in range(t0, t1 + 1):
        if sell_on is not None and t >= sell_on:
            if _sellable(c, t, j):
                p = _sellpx(c, t, j); return t, (1 - FEE) * (1 - FEE - TAX) * p / px - 1
            continue
        if not c['VALID'][t, j]:
            if t - t0 + 1 >= cap: sell_on = t + 1
            continue
        cc = c['CA'][t, j]
        if np.isnan(cc): continue
        hi = max(hi, cc)
        dd = d if (X is None or hi < px * (1 + X)) else min(d, Y)
        if cc < hi * (1 - dd) or (hi >= px * 1.06 and cc < px * 1.005) or t - t0 + 1 >= cap:
            sell_on = t + 1
    return None


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    c = sim.make_ctx('fix', P=P, F=F, sig='fix')
    sid = P['meta']['stock_id'].astype(str); odd = (sid.str[-1].astype(int) % 2 == 1).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1
    kix = {k: j for j, k in enumerate(P['meta']['security_key'])}
    R = sim.run_paths('fix', 100, sig='fix')
    seen = {}
    for i in range(100):
        for (code, d0, d1, w, ret, hold, why, st, lk) in R['X'][i][1]:
            seen.setdefault((code, d0), dict(code=code, d0=d0, d1=d1, ret=ret, why=why))
    rows = []
    for v in seen.values():
        j = kix[v['code']]; t0 = int(dates.get_loc(v['d0'])); t1 = int(dates.get_loc(v['d1']))
        if not odd[j] or t0 < t_lo or t1 > t_hi: continue
        a = c['ATRP'][t0 - 1, j] * 4; d = 0.04 if not np.isfinite(a) else min(max(a, 0.04), 0.15)
        b = replay(c, j, t0, t1, d)
        hp = c['CA'][t0:t1 + 1, j] / (c['Ov'][t0, j] * (1 + SLIP)) - 1
        rows.append(dict(j=j, t0=t0, t1=t1, d=d, ret=v['ret'], why=v['why'], rep=(b[1] if b else np.nan), rep_t=(b[0] if b else -1),
                         mfe=float(np.nanmax(hp))))
    X = pd.DataFrame(rows)
    X['組'] = np.where(X.mfe >= 0.10, np.where(X.ret < 0, 'G1', 'G2'), 'G3')
    top = X.ret >= X.ret.quantile(0.9)
    ok = X.rep_t == X.t1
    L = ['# M-103 結果：浮盈後收緊停損的交易層反事實（探索集）\n',
         f"交易 {len(X)} 筆（G1 {int((X.組 == 'G1').sum())}、G2 {int((X.組 == 'G2').sum())}、G3 {int((X.組 == 'G3').sum())}）。",
         f"重播驗證：出場日與引擎一致 {ok.mean():.1%}（不一致者多為下市出清，重播不處理下市，沿用原結果）；一致者報酬差中位數 {float((X.rep - X.ret)[ok].abs().median()):.2e}。\n"]
    res = []
    for scope in ('U', 'H'):
        for x in XS:
            for y in YS:
                dr = np.zeros(len(X)); n_aff = 0; dh = []
                for k, r in X.iterrows():
                    if scope == 'H' and r.d < 0.10: continue
                    if r.d <= y: continue                       # 新寬度不比原本窄，規則無作用
                    b = replay(c, int(r.j), int(r.t0), int(r.t1), r.d, x, y)
                    if b is None or b[0] >= r.t1: continue
                    dr[k] = b[1] - r.ret; n_aff += 1; dh.append(b[0] - r.t1)
                g = X.組.values
                res.append(dict(範圍=scope, X=x, Y=y, 影響筆數=n_aff, 總和=dr.sum(), G1=dr[g == 'G1'].sum(), G2=dr[g == 'G2'].sum(), G3=dr[g == 'G3'].sum(),
                                右尾變化=dr[top].sum() / X.ret[top].sum(), 持有日變化=float(np.mean(dh)) if dh else 0.0))
    T = pd.DataFrame(res)
    for scope, nm in (('U', 'U：全部交易'), ('H', 'H：只套用停損寬度 d ≥ 10%')):
        s = T[T.範圍 == scope]
        L += [f'\n## {nm}\n', '| X（浮盈門檻） | Y（新寬度） | 影響筆數 | 報酬變化總和 | G1 | G2 | G3 | 前 10% 交易合計變化 | 平均提早（日） |',
              '|---|---|---|---|---|---|---|---|---|']
        for _, r in s.iterrows():
            L.append(f"| {r.X:.0%} | {r.Y:.0%} | {r.影響筆數} | {r.總和:+.2f} | {r.G1:+.2f} | {r.G2:+.2f} | {r.G3:+.2f} | {r.右尾變化:+.1%} | {r.持有日變化:.0f} |")
        pos = int((s.總和 > 0).sum()); m = s[(s.X == MAIN[0]) & (s.Y == MAIN[1])].iloc[0]
        passed = pos >= 9 and m.右尾變化 >= -0.10
        L.append(f"\n門檻：總和 > 0 的組數 {pos}/12（需 ≥ 9）；主參數前 10% 交易合計變化 {m.右尾變化:+.1%}（需 ≥ −10%）→ **{'進入組合層' if passed else '不進入'}**")
        T.loc[T.範圍 == scope, 'pass'] = passed
    L.append('\n「報酬變化總和」為各筆交易報酬差的加總（等權；例如 +1.00 等於一筆交易多賺 100%）。基準總和：' + f"{X.ret.sum():+.2f}（{len(X)} 筆）。")
    pu, ph = bool(T[T.範圍 == 'U']['pass'].iloc[0]), bool(T[T.範圍 == 'H']['pass'].iloc[0])
    head = f"U 範圍{'進入' if pu else '不進入'}組合層、H 範圍{'進入' if ph else '不進入'}組合層"
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
