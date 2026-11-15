import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
Y = pd.read_csv('/home/claude/extdata/yahoo.csv', parse_dates=['Date']).set_index('Date')
def build(tk):
    d = Y[[f'{tk}_Close', f'{tk}_High', f'{tk}_Low']].dropna(); d.columns = ['C', 'H', 'L']; d = d[(d.H >= d.L) & (d.C > 0)]
    C, H, L = d.C, d.H, d.L
    ma = C.rolling(60).mean(); sl = ma / ma.shift(20) - 1; ab = (C > ma).astype(float).rolling(60).mean()
    hp, lp = H.rolling(5).max().shift(1), L.rolling(5).min().shift(1); w = H - L
    inter = (np.minimum(H, hp) - np.maximum(L, lp)).clip(lower=0); ins = ((H <= hp) & (L >= lp)).astype(float)
    ov = (inter / w.where(w > 0)).where(w > 0, ins); near = ov.rolling(5).mean()
    strong = (ab >= 0.8) & (sl > 0); P = strong & (sl < sl.shift(20)) & (near > near.shift(20)); Q = strong & ~P
    r = C.pct_change(); vol = r.rolling(20).std() * np.sqrt(252)
    c = C.values; n = len(c); dd20 = np.full(n, np.nan); dd60 = np.full(n, np.nan); f60 = np.full(n, np.nan); nh = np.full(n, np.nan); fv = np.full(n, np.nan)
    runmax = C.cummax().values; rv = r.values
    for t in range(n - 60):
        dd20[t] = c[t + 1:t + 21].min() / c[t] - 1; dd60[t] = c[t + 1:t + 61].min() / c[t] - 1; f60[t] = c[t + 60] / c[t] - 1
        nh[t] = float(c[t + 1:t + 61].max() > runmax[t]); fv[t] = np.std(rv[t + 1:t + 21]) * np.sqrt(252)
    out = pd.DataFrame({'P': P, 'Q': Q, 'vol': vol, 'dd20': dd20, 'dd60': dd60, 'f60': f60, 'nh': nh, 'fv': fv}, index=d.index).dropna()
    out['vq'] = pd.qcut(out.vol, 3, labels=['低波', '中波', '高波']); return out
def rep(X, lab):
    if len(X) < 20: return f'{lab} n{len(X)}（樣本不足）'
    return f'{lab} n{len(X):4d} ｜ dd20 {X.dd20.mean()*100:+.1f}% ｜ dd60 {X.dd60.mean()*100:+.1f}% ｜ P(dd60≤−10%) {(X.dd60<=-.1).mean()*100:4.1f}% ｜ 60日報酬 {X.f60.mean()*100:+.1f}% ｜ 未來波動 {X.fv.mean()*100:.0f}% ｜ 60日內創高 {X.nh.mean()*100:.0f}%'
def episodes(m):
    idx = np.where(m.values)[0]; return int((np.diff(np.r_[-999, idx]) > 20).sum())
T = build('^TWII')
seg = np.where(T.index.year <= 2015, '05-15', np.where(T.index.year <= 2020, '16-20', '21-26')); T = T[T.index.year >= 2005]; seg = seg[-len(T):]
print('==== ^TWII')
for s in ['05-15', '16-20', '21-26']:
    X = T[seg == s]; print(f' -- {s}（型態 P 獨立段數 {episodes(X.P)}）'); print('   ' + rep(X[X.P], '型態 P')); print('   ' + rep(X[X.Q], '強漲未減速 Q')); print('   ' + rep(X, '全部'))
print(' -- 控制當下波動（全期，波動三分位內 P vs Q 的 P(dd60≤−10%)、dd60）')
for v in ['低波', '中波', '高波']:
    for s in ['05-15', '16-20', '21-26']:
        X = T[(seg == s) & (T.vq == v)]; a, b = X[X.P], X[X.Q]
        print(f'   {v} {s} P n{len(a)} {(a.dd60<=-.1).mean()*100:4.1f}% / {a.dd60.mean()*100:+.1f}% ｜ Q n{len(b)} {(b.dd60<=-.1).mean()*100:4.1f}% / {b.dd60.mean()*100:+.1f}%')
for tk in ['^GSPC', '^N225', '^KS11', '^HSI']:
    X = build(tk); h = np.where(X.index.year <= 2014, '03-14', '15-26')
    print(f'==== {tk}')
    for s in ['03-14', '15-26']:
        Z = X[h == s]; print(f' -- {s}（P 段數 {episodes(Z.P)}）'); print('   ' + rep(Z[Z.P], '型態 P')); print('   ' + rep(Z[Z.Q], '強漲未減速 Q'))
    a, b = X[X.P], X[X.Q]; print('   控制波動（全期）：' + '；'.join(f'{v} P {(a[a.vq==v].dd60<=-.1).mean()*100:.1f}% vs Q {(b[b.vq==v].dd60<=-.1).mean()*100:.1f}%' for v in ['低波', '中波', '高波']))
