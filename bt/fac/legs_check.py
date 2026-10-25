import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from factors import base
P = load_panel(); C, H, L = P['C'], P['H'], P['L']
atrp = base(P)['atr'] / C
def fractal(h, l, N):
    piv = []
    for t in range(2 * N, len(h)):
        c = t - N; wh = h[c - N:t + 1]; wl = l[c - N:t + 1]
        if np.nanmax(wh) == h[c] and (wh == h[c]).sum() == 1:
            if piv and piv[-1][2] == 1:
                if h[c] > piv[-1][1]: piv[-1] = [c, h[c], 1]
            else: piv.append([c, h[c], 1])
        if np.nanmin(wl) == l[c] and (wl == l[c]).sum() == 1:
            if piv and piv[-1][2] == -1:
                if l[c] < piv[-1][1]: piv[-1] = [c, l[c], -1]
            else: piv.append([c, l[c], -1])
    return piv
def zz_pct(h, l, th):
    piv = []; tr = 1; ei = 0
    for t in range(1, len(h)):
        if tr == 1:
            if h[t] >= h[ei]: ei = t
            elif l[t] <= h[ei] * (1 - th): piv.append([ei, h[ei], 1]); tr = -1; ei = t
        else:
            if l[t] <= l[ei]: ei = t
            elif h[t] >= l[ei] * (1 + th): piv.append([ei, l[ei], -1]); tr = 1; ei = t
    return piv
def zz_atr(h, l, a, k):
    piv = []; tr = 1; ei = 0
    for t in range(1, len(h)):
        th = k * a[t]
        if tr == 1:
            if h[t] >= h[ei]: ei = t
            elif np.log(h[ei] / l[t]) >= th: piv.append([ei, h[ei], 1]); tr = -1; ei = t
        else:
            if l[t] <= l[ei]: ei = t
            elif np.log(h[t] / l[ei]) >= th: piv.append([ei, l[ei], -1]); tr = 1; ei = t
    return piv
def show(code, a, b, lab):
    s = slice(a, b); d = C.loc[s].index
    h, l, ap = H[code].loc[s].values, L[code].loc[s].values, atrp[code].loc[s].values
    i0 = int(np.nanargmin(l))
    for name, piv in (('3 根分形', fractal(h, l, 3)), ('5 根分形', fractal(h, l, 5)), ('8 根分形', fractal(h, l, 8)),
                      ('之字 8%', zz_pct(h, l, .08)), ('之字 2ATR', zz_atr(h, l, ap, 2)), ('之字 3ATR', zz_atr(h, l, ap, 3))):
        piv = [p for p in piv if p[0] >= i0]
        seq = [[i0, l[i0], -1]] + piv
        legs = []; last_up = None; rr = []
        for x, y in zip(seq[:-1], seq[1:]):
            ch = y[1] / x[1] - 1
            legs.append(f'{ch:+.0%}')
            if ch > 0: last_up = ch
            elif last_up: rr.append(-ch / last_up)
        print(f'{lab} {name:9s} 上漲波 {sum(1 for x in legs if x[0]=="+")}，回撤比例平均 {np.mean(rr) if rr else float("nan"):.2f}：', ' '.join(legs))
show('6223', '2025-05-20', '2025-10-09', '旺矽')
show('6215', '2024-12-01', '2025-02-18', '和椿')
show('4743', '2020-03-15', '2020-04-29', '合一')
