# M-100：X1 現行候選中的籌碼／估值剔除因子（L-110 探索集 → 提名因子才進確認集）
# 定義與判準見 lab/registry/registry_M.md 的 M-100。描述性，不計入 L 系列 DSR（n_variants＝0）。
# 用法：python lab/core/run.py M-100
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel, twdata_path
from feat import load_feat
import signals

H = 21                     # 每期間隔（交易日）；持有 t+1 開盤 → t+22 收盤
EXPLORE_END = pd.Timestamp('2015-12-31')
FACTORS = ['MGR', 'MUSE', 'INST', 'FHR', 'PBR', 'TURN', 'DISP']
DESC = dict(MGR='融資 20 日增量 ÷ 股數（高＝差）', MUSE='融資使用率（高＝差）', INST='三大法人 20 日淨買超 ÷ 成交量（低＝差）',
            FHR='外資持股比率 20 日變化（低＝差）', PBR='股價淨值比（高＝差）', TURN='20 日平均週轉率（高＝差）',
            DISP='近 21 日曾為處置股（是＝差）')


# ---------------- 籌碼資料 → 面板格子 ----------------
def staged(name, cols, years):
    d = twdata_path() / 'staged' / name
    out = [pd.read_parquet(d / f'{y}.parquet', columns=cols) for y in years if (d / f'{y}.parquet').exists()]
    df = pd.concat(out, ignore_index=True); df['date'] = pd.to_datetime(df['date'])
    return df


def to_grid(df, val, P, cols_of, dates, agg=None):
    """(stock_id, date, 值) → T×N 陣列；同代碼多段時放到當日存在（EX）的那一段。"""
    T, N = len(dates), len(P['meta'])
    A = np.full((T, N), np.nan, np.float64)
    if agg: df = df.groupby(['stock_id', 'date'], as_index=False)[val].sum()
    ti = dates.get_indexer(df['date']); keep = ti >= 0; df = df[keep]; ti = ti[keep]
    EX = np.asarray(P['EX'][:T])
    sid = df['stock_id'].to_numpy(); v = df[val].to_numpy(np.float64)
    for s, idx in pd.Series(np.arange(len(df))).groupby(sid):
        cs = cols_of.get(s)
        if not cs: continue
        rows = idx.to_numpy()
        if len(cs) == 1:
            A[ti[rows], cs[0]] = v[rows]
        else:
            for c in cs:
                m = EX[ti[rows], c]; A[ti[rows][m], c] = v[rows][m]
    return A


def ffill_limit(A, k=5):
    return pd.DataFrame(A).ffill(limit=k).to_numpy()


def build_factors(P, F, dates, years):
    M = P['meta']; cols_of = {}; Tm = len(dates)
    for j, s in enumerate(M['stock_id'].astype(str)): cols_of.setdefault(s, []).append(j)
    V = np.where(np.asarray(F['VALID'][:Tm]), np.asarray(P['V'][:Tm], np.float64), np.nan)
    mg = staged('TaiwanStockMarginPurchaseShortSale', ['date', 'stock_id', 'MarginPurchaseTodayBalance', 'MarginPurchaseLimit'], years)
    BAL = ffill_limit(to_grid(mg, 'MarginPurchaseTodayBalance', P, cols_of, dates))
    LIM = ffill_limit(to_grid(mg, 'MarginPurchaseLimit', P, cols_of, dates)); del mg
    sh = staged('TaiwanStockShareholding', ['date', 'stock_id', 'NumberOfSharesIssued', 'ForeignInvestmentSharesRatio'], years)
    SHR = ffill_limit(to_grid(sh, 'NumberOfSharesIssued', P, cols_of, dates))
    FR = ffill_limit(to_grid(sh, 'ForeignInvestmentSharesRatio', P, cols_of, dates)); del sh
    it = staged('TaiwanStockInstitutionalInvestorsBuySell', ['date', 'stock_id', 'buy', 'sell'], years)
    it['net'] = it['buy'].astype(float) - it['sell'].astype(float)
    NET = to_grid(it, 'net', P, cols_of, dates, agg=True); del it
    pe = staged('TaiwanStockPER', ['date', 'stock_id', 'PBR'], years)
    PB = ffill_limit(to_grid(pe, 'PBR', P, cols_of, dates)); del pe
    SHR = np.where(SHR > 0, SHR, np.nan); PB = np.where(PB > 0, PB, np.nan)
    lag = lambda A, k=20: np.vstack([np.full((k, A.shape[1]), np.nan), A[:-k]])
    roll = lambda A: pd.DataFrame(A).rolling(20, min_periods=15).sum().to_numpy()
    n_inst = pd.DataFrame(np.isfinite(NET)).rolling(20).sum().to_numpy()
    with np.errstate(all='ignore'):
        X = dict(
            MGR=(BAL - lag(BAL)) * 1000 / SHR,
            MUSE=np.where(LIM > 0, BAL / LIM, np.nan),
            INST=-np.where(n_inst >= 15, pd.DataFrame(np.nan_to_num(NET)).rolling(20).sum().to_numpy() / roll(V), np.nan),
            FHR=-(FR - lag(FR)),
            PBR=PB,
            TURN=pd.DataFrame(V).rolling(20, min_periods=15).mean().to_numpy() / SHR,
        )
    D = np.asarray(P['DISP'][:Tm], bool)
    X['DISP'] = pd.DataFrame(D).rolling(21, min_periods=1).max().to_numpy().astype(bool)
    return X


# ---------------- 統計 ----------------
def nw_t(x, lag=2):
    x = np.asarray(x, float); n = len(x)
    if n < 5: return float('nan')
    xc = x - x.mean(); v = (xc @ xc) / n
    for l in range(1, lag + 1): v += 2 * (1 - l / (lag + 1)) * (xc[l:] @ xc[:-l]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else float('nan')


def periods(dates):
    t0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
    ts = []
    t = t0
    while t + 22 < len(dates) and dates[t + 22] <= EXPLORE_END:
        ts.append(t); t += H
    return ts


def returns_at(t, CA, OA, VALID, OSUS):
    """t+1 開盤進場、t+22 收盤（停牌／下市取最後有效收盤）出場。回傳（報酬, 可買）"""
    ent = OA[t + 1].astype(float); ok = VALID[t + 1] & ~OSUS[t + 1] & np.isfinite(ent) & (ent > 0)
    win = CA[t + 1:t + 23].astype(float)
    ex = pd.DataFrame(win).ffill().to_numpy()[-1]
    with np.errstate(all='ignore'): r = ex / ent - 1
    return r, ok & np.isfinite(r)


def evaluate(X, scope, mask_cols, ts, R, OK, names):
    """scope：T×N 候選；mask_cols：探索／確認的股票欄位。回傳每個因子的彙總 dict。"""
    res = {}
    per = {f: [] for f in names}
    for k, t in enumerate(ts):
        S = scope[t] & mask_cols & OK[k]
        if S.sum() < 10: continue
        ex = R[k][S] - R[k][S].mean()
        q10, q90 = np.quantile(ex, [0.1, 0.9])
        for f in names:
            x = X[f][t][S]
            if f == 'DISP':
                fin = np.ones(len(x), bool); flag = x.astype(bool)
            else:
                fin = np.isfinite(x)
                if fin.sum() < 10: continue
                pr = pd.Series(np.where(fin, x, np.nan)).rank(pct=True).to_numpy()
                flag = fin & (pr > 0.8)
            keep = fin & ~flag
            if flag.sum() < 1 or keep.sum() < 1: continue
            per[f].append(dict(t=t, d=ex[keep].mean() - ex[flag].mean(), nf=int(flag.sum()), nk=int(keep.sum()),
                               nS=int(S.sum()), cov=float(fin.mean()), n_left=int((ex[flag] <= q10).sum()),
                               n_right=int((ex[flag] >= q90).sum())))
    for f in names:
        df = pd.DataFrame(per[f])
        if df.empty:
            res[f] = dict(n=0); continue
        v = df[df.nf >= 5]
        res[f] = dict(n=len(df), n_valid=len(v), d=float(v.d.mean()) if len(v) else np.nan, t=nw_t(v.d) if len(v) else np.nan,
                      ann=float(v.d.mean() * 12) if len(v) else np.nan, nf=float(df.nf.mean()), nS=float(df.nS.mean()),
                      cov=float(df['cov'].mean()), left=float(v['n_left'].sum() / v['nf'].sum()) if len(v) else np.nan,
                      right=float(v['n_right'].sum() / v['nf'].sum()) if len(v) else np.nan, flags=int(v.nf.sum()),
                      first=str(pd.Timestamp(PDATES[int(df.t.min())]).date()))
    return res


def verdict(r):
    ok = (r.get('n_valid', 0) >= 60 and r['d'] > 0 and r['t'] >= 2.5 and r['right'] <= 0.10 and r['left'] >= 0.125)
    why = []
    if r.get('n_valid', 0) < 60: why.append(f"有效期數 {r.get('n_valid', 0)} < 60")
    if not (r.get('d', 0) > 0 and r.get('t', 0) >= 2.5): why.append('t < 2.5' if r.get('d', 0) > 0 else 'd ≤ 0')
    if r.get('right', 1) > 0.10: why.append('右尾 > 10%')
    if r.get('left', 0) < 0.125: why.append('左尾 < 12.5%')
    return ok, '、'.join(why)


def fmt(res, names, crit=True):
    rows = ['| 因子 | 期數（有效） | 平均候選／標記 | 涵蓋 | 保留−標記（每 21 日） | 年化 | NW t | 左尾率 | 右尾率 | ' + ('判定 |' if crit else ''),
            '|---|---|---|---|---|---|---|---|---|' + ('---|' if crit else '')]
    for f in names:
        r = res[f]
        if not r.get('n'):
            rows.append(f'| {f} | 0 | — | — | — | — | — | — | — |' + (' 無資料 |' if crit else '')); continue
        line = (f"| {f} | {r['n']}（{r['n_valid']}） | {r['nS']:.0f}／{r['nf']:.1f} | {r['cov']:.0%} | {r['d']*100:+.2f}% | "
                f"{r['ann']*100:+.1f}% | {r['t']:.2f} | {r['left']:.1%} | {r['right']:.1%} |")
        if crit:
            ok, why = verdict(r); line += (' **提名** |' if ok else f' 不過（{why}） |')
        rows.append(line)
    return '\n'.join(rows)


PDATES = None


def md_table(df):
    out = ['| | ' + ' | '.join(df.columns) + ' |', '|---' * (len(df.columns) + 1) + '|']
    for i, row in df.iterrows():
        out.append(f'| {i} | ' + ' | '.join('—' if not np.isfinite(v) else f'{v:+.2f}' for v in row.values) + ' |')
    return '\n'.join(out)


def main():
    global PDATES
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; PDATES = dates
    sig = signals.compute('fix', P=P, F=F); dbg = sig['_dbg']
    Tm = int(dates.searchsorted(EXPLORE_END)) + 30            # 只需要到 2015 年底（加一點緩衝），省記憶體
    pool, hot, steep, Tr = (np.asarray(dbg[k][:Tm]) for k in ('pool', 'hot', 'steep', 'Tr'))
    del sig, dbg
    candX = pool & ~hot & ~steep
    dates = dates[:Tm]
    sid = P['meta']['stock_id'].astype(str)
    last = sid.str[-1].astype(int).to_numpy()
    odd, even = (last % 2 == 1), (last % 2 == 0)
    ts = periods(dates)
    years = list(range(2004, EXPLORE_END.year + 1))
    X = build_factors(P, F, dates, years)
    CA, OA = np.asarray(F['CA'][:Tm]), np.asarray(F['OA'][:Tm]); VALID = np.asarray(F['VALID'][:Tm]); OSUS = np.asarray(P['OSUS'][:Tm])
    R, OK = [], []
    for t in ts:
        r, ok = returns_at(t, CA, OA, VALID, OSUS); R.append(r); OK.append(ok)

    exp_c = evaluate(X, candX, odd, ts, R, OK, FACTORS)
    exp_p = evaluate(X, pool, odd, ts, R, OK, FACTORS)
    nominated = [f for f in FACTORS if exp_c[f].get('n') and verdict(exp_c[f])[0]]
    conf = evaluate(X, candX, even, ts, R, OK, nominated) if nominated else {}

    # 池內相關（探索集候選，各期 Spearman 平均）
    R20 = np.asarray(F['R20'][:Tm]); S60 = np.asarray(F['S60'][:Tm])
    ref = dict(R20=R20, S60=S60, TLD=Tr)
    allv = FACTORS[:-1] + list(ref)
    cm = []
    for t in ts:
        S = candX[t] & odd
        if S.sum() < 15: continue
        df = pd.DataFrame({k: (X[k][t][S] if k in X else ref[k][t][S]) for k in allv})
        cm.append(df.rank().corr().values)
    cm = pd.DataFrame(np.nanmean(np.array(cm), axis=0), index=allv, columns=allv)

    L = [f"# M-100 結果：X1 現行候選中的籌碼／估值剔除因子\n",
         f"期數：{len(ts)}（訊號日 {dates[ts[0]].date()} ～ {dates[ts[-1]].date()}，每 {H} 個交易日；出場日不晚於 {EXPLORE_END.date()}）。"
         f"報酬 t+1 開盤 → t+22 收盤（還原價）；超額＝扣同期候選等權平均。因子方向已調成「越高越差」，標記＝候選內最差 20%（DISP 為二元）。\n",
         "因子：" + '；'.join(f'{k} {v}' for k, v in DESC.items()) + "。\n",
         "## 探索集（奇數代碼）— X1 現行候選（判準適用）\n", fmt(exp_c, FACTORS),
         "\n判準：有效期數 ≥ 60（每期標記 ≥ 5）、保留−標記 > 0 且 t ≥ 2.5、右尾率 ≤ 10%、左尾率 ≥ 12.5%。隨機標記時左右尾率的期望值都是 10%。\n",
         "## 探索集 — 全強勢池（只報告）\n", fmt(exp_p, FACTORS, crit=False), "\n",
         "## 因子相關（探索集候選內，各期 Spearman 平均）\n", md_table(cm), "\n"]
    if nominated:
        L += ["## 確認集（偶數代碼，只測提名因子一次）\n", fmt(conf, nominated, crit=False)]
        for f in nominated:
            r = conf[f]; ok = r.get('n_valid', 0) > 0 and r['d'] > 0 and r['t'] >= 2.0 and r['right'] <= 0.10
            L.append(f"- {f}：{'**確認通過**' if ok else '確認不通過（結案）'}（d {r['d']*100:+.2f}%、t {r['t']:.2f}、右尾 {r['right']:.1%}）")
    else:
        L.append("## 確認集\n沒有提名因子，不執行確認集（偶數代碼未被使用）。")
    head = f"提名 {len(nominated)} 個：{'、'.join(nominated) if nominated else '無'}"
    if nominated:
        passed = [f for f in nominated if conf[f].get('n_valid', 0) > 0 and conf[f]['d'] > 0 and conf[f]['t'] >= 2.0 and conf[f]['right'] <= 0.10]
        head += f"；確認通過：{'、'.join(passed) if passed else '無'}"
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
