# M-154：V4.1 事件的區分因子掃描（23 個單因子＋6 個事前設計聯合因子）。觀察式，探索集（奇數代碼、2005–15）。定義見 registry_M.md M-154。
# 產出：lab/cache/M154/plot.json（作圖用彙總，不入庫）＋ summary（run.py 寫入 lab/results/M154/）。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from chips import crowd

START = pd.Timestamp('2005-01-01'); END = pd.Timestamp('2015-12-31'); H = 20
SPLIT = 2010
OUT = LAB / 'cache' / 'M154'

NAMES = {
    'A1': '收盤距 MA25', 'A2': '收盤距 MA99', 'A3': 'R20 20 日漲幅', 'A4': 'S60 長均線斜率', 'A5': 'RRT 60 日回撤比',
    'A6': 'T_ld 池內百分位', 'A7': 'ΔT_ld10 池內百分位', 'A8': 'ATR%', 'A9': '事件日漲幅', 'A10': '收盤位置',
    'A11': '上影線比', 'A12': '量比', 'A13': '含整理突破（EV2）', 'A14': '整理寬度（前 60 日）', 'A15': 'log 成交額',
    'A16': '20 日週轉率', 'A17': '融資使用率', 'A18': 'R4 營收創高', 'A19': 'R5 年增驚喜', 'A20': '指數 60 日報酬',
    'A21': '指數距 200 日線', 'A22': 'V4.1 近 20 日訊號數', 'A23': '隔日開盤跳空（進場時）',
    'B1': 'B1 過度延伸', 'B2': 'B2 擁擠', 'B3': 'B3 健康突破（條件式）', 'B4': 'B4 緊整理突破', 'B5': 'B5 基本面加持', 'B6': 'B6 動能品質'}
JOINT = {'B1': [('A1', 1), ('A3', 1), ('A8', 1)], 'B2': [('A16', 1), ('A17', 1)], 'B4': [('A14', -1), ('A2', -1)],
         'B5': [('A18', 1), ('A19', 1)], 'B6': [('A6', 1), ('A1', -1)]}


def rev_month_factors():
    # 同 M120.rev_factors 的 R4、R5；period_end 在 pyarrow 環境讀成字串，先轉日期
    import M120
    d = pd.read_parquet(M120.TWD / 'derived' / 'month_revenue.parquet')
    d = d[d.security_key.notna() & d.security_key.ne('')].copy()
    d['m'] = pd.to_datetime(d.period_end).dt.to_period('M')
    W = d.pivot_table(index='m', columns='security_key', values='revenue', aggfunc='last').astype(float)
    W = W.reindex(pd.period_range(W.index.min(), W.index.max(), freq='M')); W[W <= 0] = np.nan
    s3 = W.rolling(3, min_periods=3).sum()
    with np.errstate(all='ignore'):
        R1 = W / W.shift(12) - 1
        R4 = s3 / s3.shift(1).rolling(24, min_periods=12).max() - 1
        mu = R1.shift(1).rolling(12, min_periods=10).mean(); sd = R1.shift(1).rolling(12, min_periods=10).std()
        R5 = (R1 - mu) / sd.where(sd > 0)
    return R4, R5


def ic(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20: return np.nan, np.nan, int(ok.sum())
    r = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
    n = ok.sum(); t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(r), float(t), int(n)


def stat(y, mae):
    y = y[np.isfinite(y)]
    if len(y) == 0: return dict(n=0)
    return dict(n=int(len(y)), mean=float(y.mean()), med=float(np.median(y)), win=float((y > 0).mean()),
                succ=float((y >= .05).mean()), fake=float((y <= -.05).mean()), lt=float((y <= -.10).mean()),
                rt=float((y >= .15).mean()), sd=float(y.std()), mae=float(np.nanmedian(mae)) if len(mae) else np.nan)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; daily = S['daily']
    U, signal, Tr = dbg['U'], dbg['signal'], dbg['Tr']
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); V = np.asarray(P['V'], np.float64)
    ATR = np.asarray(F['ATR']); R20 = np.asarray(F['R20']); S60 = np.asarray(F['S60']); RRT = np.asarray(F['RRT'])
    AMT = np.asarray(F['AMT20V']); EV2 = np.asarray(F['EV2'])
    TURN, MUSE = crowd(P, F, 20)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CEW = np.cumprod(1 + ew)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    ix60 = ix.pct_change(60, fill_method=None).to_numpy(); ixma = (ix / ix.rolling(200).mean() - 1).to_numpy()
    br20 = daily['n_signal'].rolling(20, min_periods=1).sum().to_numpy()
    sid = P['meta']['stock_id'].astype(str); odd = (sid.str[-1].astype(int) % 2 == 1).to_numpy()
    skey = P['meta']['security_key'].to_numpy()
    R4, R5 = rev_month_factors()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1

    tt, jj = np.nonzero(signal)
    ok = (tt >= t_lo) & (tt + H <= t_hi) & odd[jj]; tt, jj = tt[ok], jj[ok]
    # 池內百分位（只算事件日）
    pr6, pr7 = {}, {}
    for t in np.unique(tt):
        u = U[t]
        a = pd.Series(np.where(u, Tr[t], np.nan)).rank(pct=True).to_numpy()
        d = pd.Series(np.where(u, Tr[t] - Tr[t - 10], np.nan)).rank(pct=True).to_numpy()
        for j in jj[tt == t]: pr6[(t, j)] = a[j]; pr7[(t, j)] = d[j]
    rows = []
    for j in np.unique(jj):
        ii = np.flatnonzero(VALID[:, j]); pos = {t: k for k, t in enumerate(ii)}
        c, h, l, v = CA[ii, j], HA[ii, j], LA[ii, j], V[ii, j]; o = OA[ii, j]
        s = pd.Series(c); m25 = s.rolling(25).mean().to_numpy(); m99 = s.rolling(99).mean().to_numpy()
        vma = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        hh60 = pd.Series(h).rolling(60).max().shift(1).to_numpy(); ll60 = pd.Series(l).rolling(60).min().shift(1).to_numpy()
        for t in tt[jj == j]:
            k = pos.get(t)
            if k is None or k < 100: continue
            ent = OA[t + 1, j]
            if not (np.isfinite(ent) and ent > 0 and VALID[t + 1, j]): continue
            rng_ = h[k] - l[k]
            y = Cf[t + H, j] / ent - 1 - (CEW[t + H] / CEW[t] - 1)
            seg = LA[t + 1:t + H + 1, j]; mae = np.nanmin(seg) / ent - 1 if np.isfinite(seg).any() else np.nan
            per = dates[t].to_period('M') - (1 if dates[t].day > 10 else 2)
            def rv(Wf):
                try: return float(Wf.at[per, skey[j]])
                except KeyError: return np.nan
            rows.append(dict(t=int(t), j=int(j), year=int(dates[t].year), Y=float(y), MAE=float(mae),
                             Yc=float(Cf[t + H, j] / Cf[t, j] - 1 - (CEW[t + H] / CEW[t] - 1)),
                             A1=c[k] / m25[k] - 1, A2=c[k] / m99[k] - 1, A3=float(R20[t, j]), A4=float(S60[t, j]), A5=float(RRT[t, j]),
                             A6=pr6[(t, j)], A7=pr7[(t, j)], A8=float(ATR[t, j] / CA[t, j]), A9=c[k] / c[k - 1] - 1,
                             A10=(c[k] - l[k]) / rng_ if rng_ > 0 else np.nan,
                             A11=(h[k] - max(o[k], c[k])) / rng_ if rng_ > 0 else np.nan,
                             A12=v[k] / vma[k] if vma[k] > 0 else np.nan, A13=float(bool(EV2[t, j])), A14=hh60[k] / ll60[k] - 1,
                             A15=float(np.log(AMT[t, j])) if AMT[t, j] > 0 else np.nan, A16=float(TURN[t, j]), A17=float(MUSE[t, j]),
                             A18=rv(R4), A19=rv(R5), A20=float(ix60[t]), A21=float(ixma[t]), A22=float(br20[t]),
                             A23=OA[t + 1, j] / CA[t, j] - 1))
    X = pd.DataFrame(rows)
    pr = lambda col: X[col].rank(pct=True)
    for b, comp in JOINT.items():
        X[b] = pd.concat([pr(a) if sgn > 0 else 1 - pr(a) for a, sgn in comp], axis=1).mean(axis=1, skipna=False)
    X['B3'] = ((X.A10 >= 0.7) & X.A12.between(1.5, 4) & (X.A1 <= X.A1.median())).astype(float)

    Y = X.Y.to_numpy(); MAE = X.MAE.to_numpy(); e1 = (X.year <= SPLIT).to_numpy()
    base = stat(Y, MAE)
    res = []
    for f in [f'A{i}' for i in range(1, 24)] + ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
        x = X[f].to_numpy(float)
        r, t, n = ic(x, Y); r1, t1, _ = ic(np.where(e1, x, np.nan), Y); r2, t2, _ = ic(np.where(~e1, x, np.nan), Y)
        yrs = [ic(np.where(X.year == yy, x, np.nan), Y)[0] for yy in range(2005, 2016)]
        pos_years = int(np.sum(np.array(yrs) > 0)); n_years = int(np.sum(np.isfinite(yrs)))
        binary = f in ('A13', 'B3')
        if binary:
            grp = [stat(Y[x == 0], MAE[x == 0]), stat(Y[x == 1], MAE[x == 1])]
            a, b = grp[1], grp[0]
            tw = (a['mean'] - b['mean']) / np.sqrt(a['sd'] ** 2 / a['n'] + b['sd'] ** 2 / b['n'])
            s1 = Y[e1 & (x == 1)].mean() - Y[e1 & (x == 0)].mean(); s2 = Y[~e1 & (x == 1)].mean() - Y[~e1 & (x == 0)].mean()
            spread = a['mean'] - b['mean']; lo_hi = None
            nominate = abs(tw) >= 2.5 and np.sign(s1) == np.sign(s2) == np.sign(spread)
            tstat = float(tw)
        else:
            okx = np.isfinite(x)
            q = pd.qcut(pd.Series(x[okx]).rank(method='first'), 5, labels=False).to_numpy()
            grp = [stat(Y[okx][q == g], MAE[okx][q == g]) for g in range(5)]
            edges = [float(np.nanpercentile(x, p)) for p in (0, 20, 40, 60, 80, 100)]
            spread = grp[4]['mean'] - grp[0]['mean']; lo_hi = edges
            cut_ok = True
            if r < 0:   # 剔除方向：高分組（Q5）右尾率不得高於其他組平均
                cut_ok = grp[4]['rt'] <= np.mean([g['rt'] for g in grp[:4]])
            else:       # 正向：低分組（Q1）被剔，Q1 右尾率不得高於其他組平均
                cut_ok = grp[0]['rt'] <= np.mean([g['rt'] for g in grp[1:]])
            nominate = abs(t) >= 2.5 and np.sign(r1) == np.sign(r2) == np.sign(r) and np.sign(spread) == np.sign(r) and cut_ok
            tstat = float(t)
        c1 = pd.Series(x).rank().corr(X.A1.rank()); c6 = pd.Series(x).rank().corr(X.A6.rank())
        res.append(dict(f=f, name=NAMES[f], ic=r, t=tstat, n=n, ic1=r1, ic2=r2, pos_years=pos_years, n_years=n_years,
                        grp=grp, edges=lo_hi, spread=float(spread), nominate=bool(nominate), binary=binary,
                        corrA1=float(c1), corrA6=float(c6), cover=float(np.isfinite(x).mean())))

    # 附加揭露（非判準）：日期層因子以「每日平均 Y」做 IC，避免同日事件重複計數
    dl = {}
    for f in ('A20', 'A21', 'A22'):
        g = X.groupby('t').agg(x=(f, 'first'), y=('Y', 'mean'))
        r_, t_, n_ = ic(g.x.to_numpy(float), g.y.to_numpy(float)); dl[f] = dict(ic=r_, t=t_, n=n_)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plot.json').write_text(json.dumps(dict(base=base, res=res, n=int(len(X)), daylevel=dl), ensure_ascii=False, default=float), encoding='utf-8')

    pc = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:+.2%}'
    pp = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:.0%}'
    f2 = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:+.3f}'
    L = ['# M-154 結果：V4.1 事件區分因子掃描（觀察式，探索集）\n',
         f"奇數代碼、2005–15；事件 {len(X)}。主結果 Y＝t+1 開盤進、t+20 收盤的超額。全部：平均 {pc(base['mean'])}、中位數 {pc(base['med'])}、勝率 {pp(base['win'])}、成功率 {pp(base['succ'])}、假突破率 {pp(base['fake'])}、MAE 中位數 {pc(base['mae'])}。\n",
         '## 因子總表（依 |t| 排序）\n',
         '| 因子 | IC | t | 05–10 IC | 11–15 IC | IC>0 年數 | Q1 平均 | Q5 平均 | Q5−Q1 | Q1 假突破 | Q5 假突破 | Q1 右尾 | Q5 右尾 | 與 A1 相關 | 與 A6 相關 | 覆蓋 | 提名 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(res, key=lambda r: -abs(r['t']) if np.isfinite(r['t']) else 0):
        g = r['grp']; lo, hi = g[0], g[-1]
        L.append(f"| {r['name']} | {f2(r['ic'])} | {r['t']:+.1f} | {f2(r['ic1'])} | {f2(r['ic2'])} | {r['pos_years']}/{r['n_years']} | {pc(lo['mean'])} | {pc(hi['mean'])} | {pc(r['spread'])} "
                 f"| {pp(lo['fake'])} | {pp(hi['fake'])} | {pp(lo['rt'])} | {pp(hi['rt'])} | {r['corrA1']:+.2f} | {r['corrA6']:+.2f} | {r['cover']:.0%} | {'**是**' if r['nominate'] else ''} |")
    L.append('\n（A13、B3 為二元：Q1＝否、Q5＝是，t 為 Welch t。）\n')
    L += ['## 附加揭露：日期層因子以每日平均計算（非判準）\n', '| 因子 | 日數 | IC | t |', '|---|---|---|---|']
    for f, d in dl.items(): L.append(f"| {NAMES[f]} | {d['n']} | {f2(d['ic'])} | {d['t']:+.1f} |")
    L.append('')
    L += ['## 五分組明細（平均 Y ／ 假突破率 ／ 右尾率 ／ MAE 中位數）\n', '| 因子 | Q1 | Q2 | Q3 | Q4 | Q5 |', '|---|---|---|---|---|---|']
    for r in res:
        cells = [f"{pc(g['mean'])} ／ {pp(g['fake'])} ／ {pp(g['rt'])} ／ {pc(g['mae'])} (n {g['n']})" for g in r['grp']]
        if r['binary']: cells = [cells[0], '', '', '', cells[1]]
        L.append(f"| {r['name']} | " + ' | '.join(cells) + ' |')
    nom = [r['name'] for r in res if r['nominate']]
    return dict(summary='\n'.join(L), headline=f"V4.1 事件因子掃描：提名 {len(nom)} 個（{'、'.join(nom) or '無'}）", n_variants=0)
