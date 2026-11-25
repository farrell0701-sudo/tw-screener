# M-185：早期失敗辨識出場訊號 ③ 爆量長黑跌破低點（k=1/2/3）、④ 連跌＋短期重疊率走低（k=2/3/4）（奇數代碼）。定義見 registry_M.md M-185。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
import M182
from panel import load as load_panel
from feat import load_feat

ERAS = M182.ERAS; COST = 0.0077; HOLD = 120
K3 = (1, 2, 3); K4 = (2, 3, 4)


class Stock:
    """單一股票在有效交易日序列上的資料與指標（缺漏日跳過）。"""
    def __init__(self, j, CA, OA, HA, LA, AMT, VALID):
        v = np.nonzero(VALID[:, j] & np.isfinite(CA[:, j]))[0]; self.v = v
        self.c, self.o, self.h, self.l = CA[v, j], OA[v, j], HA[v, j], LA[v, j]
        self.o = np.where(np.isfinite(self.o), self.o, self.c)
        a = pd.Series(AMT[v, j].astype(float))
        self.a = a.to_numpy(); self.a95 = a.shift(1).rolling(120, min_periods=60).quantile(0.95).to_numpy()
        lr = pd.Series(self.l).shift(1).rolling(20, min_periods=20).min().to_numpy()
        hr = pd.Series(self.h).shift(1).rolling(20, min_periods=20).max().to_numpy()
        with np.errstate(all='ignore'):
            inter = np.clip(np.minimum(self.h, hr) - np.maximum(self.l, lr), 0, None)
            uni = np.maximum(self.h, hr) - np.minimum(self.l, lr)
            o = np.where(uni > 0, inter / uni, np.nan)
        self.ofs = pd.Series(o).rolling(5, min_periods=5).mean().to_numpy()
        self.down = np.r_[False, self.c[1:] < self.c[:-1]]
        self.ofd = np.r_[False, self.ofs[1:] < self.ofs[:-1]]


def simulate(S, t):
    """回傳 S15 與各訊號版本的 (淨報酬, 出場位置, 原因)。位置為有效日序列索引。"""
    v = S.v; e = np.searchsorted(v, t + 1)
    if e >= len(v) or v[e] > t + 5: return None
    end = np.searchsorted(v, t + HOLD, side='right') - 1
    if end <= e + 1: return None
    ent = S.o[e]; c = S.c
    # S15
    st = None
    for i in range(e, end + 1):
        if c[i] < ent * 0.85: st = i; break
    def px_after(i):  # 訊號日 i 的次一有效日開盤；若已到尾端，用 i 收盤
        return (S.o[i + 1], i + 1) if i + 1 <= end else (c[i], i)
    if st is not None: p15, x15 = px_after(st)
    else: p15, x15 = c[end], end
    base = dict(net=p15 / ent - 1 - COST, stop=st is not None, stop_i=None if st is None else st - e)
    out = {'S15': base}
    # ③ 觸發日（各 k）
    trig3 = {k: None for k in K3}
    runmax = c[e]; lastnh = None
    cands = []
    for i in range(e + 1, end + 1):
        if c[i] > runmax: runmax = c[i]; lastnh = i
        if lastnh is not None and 1 <= i - lastnh <= 10 and S.down[i] and np.isfinite(S.a95[i]) and S.a[i] >= S.a95[i]:
            cands.append(i)
    for b in cands:
        run = 0
        for i in range(b + 1, end + 1):
            if c[i] < S.l[b]:
                run += 1
                for k in K3:
                    if run == k and (trig3[k] is None or i < trig3[k]): trig3[k] = i
                if run >= max(K3): break
            else: break
    # ④
    trig4 = {k: None for k in K4}
    run = 0
    for i in range(e, end + 1):
        run = run + 1 if (S.down[i] and S.ofd[i]) else 0
        for k in K4:
            if trig4[k] is None and run >= k and i >= e + 1: trig4[k] = i
    for nm, tr in [(f'③ k={k}', trig3[k]) for k in K3] + [(f'④ k={k}', trig4[k]) for k in K4]:
        if tr is not None and (st is None or tr < st):
            p, x = px_after(tr); out[nm] = dict(net=p / ent - 1 - COST, sig=True, sig_i=tr - e)
        else:
            out[nm] = dict(net=base['net'], sig=False, sig_i=None)
        out[nm]['stop_after'] = st is not None and (tr is None or tr >= st)
    return out


def taiex_below(dates):
    d = pd.read_parquet(Path('/home/claude/tw-data-v1.1') / 'derived' / 'index.parquet')
    d = d[d.index_id == 'TAIEX']; d['date'] = pd.to_datetime(d.date)
    c = d.set_index('date').price_index.reindex(dates).ffill()
    mon = c.index.to_period('M'); below = pd.Series(False, index=dates)
    vals = c.to_numpy()
    for k in range(1, len(c)):
        s0 = c.index.searchsorted((mon[k] - 1).start_time)
        w = vals[s0:k]
        if len(w) < 10 or not np.isfinite(vals[k]): continue
        below.iloc[k] = vals[k] < (np.nanmax(w) + np.nanmin(w)) / 2
    return below.to_numpy()


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    CA = pd.DataFrame(np.asarray(F['CA'], np.float64)).to_numpy(); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); VALID = np.asarray(F['VALID'])
    AMT = np.asarray(P['AMT'], np.float64)
    D = M182.build(parity=1); D = D[D.q >= 2].reset_index(drop=True)
    below = taiex_below(dates)
    cache = {}; rows = []
    for r in D.itertuples():
        S = cache.get(r.j) or cache.setdefault(r.j, Stock(r.j, CA, OA, HA, LA, AMT, VALID))
        o = simulate(S, r.t)
        if o is None: continue
        sb = None
        if o['S15']['stop']:
            sb = bool(below[S.v[np.searchsorted(S.v, r.t + 1) + o['S15']['stop_i']]])
        for nm, x in o.items():
            rows.append(dict(date=r.date, v=nm, net=x['net'], sig=x.get('sig', False), sig_i=x.get('sig_i'),
                             base=o['S15']['net'], stop=o['S15']['stop'], stop_i=o['S15']['stop_i'], mkt_below=sb))
    R = pd.DataFrame(rows)
    VS = ['S15'] + [f'③ k={k}' for k in K3] + [f'④ k={k}' for k in K4]
    L = ['# M-185 結果：早期失敗辨識出場（R6 頭部剔除 Q1；奇數代碼；F 計數 62）\n',
         '進場 t+1 開盤；共同規則 15% 硬停損、最長 120 日；訊號次一有效日開盤出場；成本 0.77%。準確率＝被訊號出場者中，S15 下淨報酬 ≤ 0 的比例。\n']
    res = {}
    for era, lo, hi in ERAS:
        E = R[(R.date >= lo) & (R.date <= hi)]
        b = E[E.v == 'S15']
        L += [f'## {era}（{len(b):,} 筆；S15 輸家比例 {(b.net <= 0).mean():.0%}）\n',
              '| 版本 | 觸發率 | 準確率 | 觸發日（中位） | 比 S15 停損早（中位天數） | 平均 | 中位數 | 勝率 | P5 | 被訊號出場者：出場淨報酬中位 vs 續抱（S15）中位 |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for nm in VS:
            g = E[E.v == nm]; sg = g[g.sig]
            prec = (sg.base <= 0).mean() if len(sg) else np.nan
            both = sg[sg.stop & sg.sig_i.notna() & sg.stop_i.notna()]
            early = (both.stop_i - both.sig_i).median() if len(both) else np.nan
            res[(nm, era)] = dict(prec=prec, med=g.net.median(), p5=g.net.quantile(.05))
            L.append(f"| {nm} | {g.sig.mean():.1%} | {'' if nm == 'S15' else f'{prec:.0%}'} | {'' if nm == 'S15' else f'{sg.sig_i.median():.0f}'} | "
                     f"{'' if nm == 'S15' or not np.isfinite(early) else f'{early:.0f}'} | {g.net.mean():+.2%} | {g.net.median():+.2%} | {(g.net > 0).mean():.1%} | {g.net.quantile(.05):+.1%} | "
                     f"{'' if nm == 'S15' or not len(sg) else f'{sg.net.median():+.1%} vs {sg.base.median():+.1%}'} |")
        st = b[b.stop]
        L += ['', f"S15 停損 {len(st):,} 筆，停損日大盤在大盤多空點之下 {st.mkt_below.mean():.0%}（全期間交易日在其下的比例約 {np.mean(below):.0%}；加權指數以收盤近似高低）。\n"]
    vt = []
    for fam, ks, mid in (('③', K3, 2), ('④', K4, 3)):
        ok = {}
        for k in ks:
            nm = f'{fam} k={k}'; a = [res[(nm, e)] for e, _, _ in ERAS]; b0 = [res[('S15', e)] for e, _, _ in ERAS]
            ok[k] = all(x['prec'] >= 0.55 and x['med'] > y['med'] and x['p5'] >= y['p5'] - 0.005 for x, y in zip(a, b0))
        nom = ok[mid] and sum(ok.values()) >= 2
        vt.append(f"- {fam}：" + '、'.join(f"k={k} {'✓' if ok[k] else '✗'}" for k in ks) + f" → {'提名（主版本 k=' + str(mid) + '）' if nom else '不提名'}")
    head = ['## 判準（準確率 ≥ 55%、中位數 > S15、P5 不深於 S15 −0.5pt；兩期；3 個 k 中 ≥ 2 個且含中間值）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline='；'.join(x[2:] for x in vt), n_variants=0)
