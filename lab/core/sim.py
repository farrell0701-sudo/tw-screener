# 模擬器：runX（X1）、runV（V4.1）、mix（合併每月再平衡）、run_paths（擾動路徑，多程序）
# 相容模式（bias3＝True）逐行對應 lab/finlab/fl_small_report.py 第 158–245 行（對照測試 lab/core/test_sim.py）。
# 修正模式（bias3＝False）依 L-113（第 3 點、第 3 點修訂、各補充）：
#   - 只在有效日（有成交、未停牌、非興櫃）判斷出場、成交；停牌中以最後價估值、佔名額。
#   - 250 日上限／持有期滿／IAKF 在停牌中到期 → 恢復交易後次一有效開盤賣。
#   - 開盤價可疑日不買；賣出改用當日收盤。漲跌停鎖死用 tw-data 旗標。
#   - 最後有效日之後不再於上市櫃交易者：delist='A' 於最後有效日收盤出清；
#     delist=('B', 折扣)：最後有效日跌停鎖死者改為凍結，估值＝最後價×(1−折扣)，資金在下市日（無則永不）回收；其餘同 A。
# 成本：手續費每邊 0.085%、證交稅 0.3%、滑價每邊 0.15%；1 倍。
import sys, os, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
FEE, TAX, SLIP = 0.00085, 0.003, 0.0015
START = pd.Timestamp('2005-01-01')


def make_ctx(mode='compat', P=None, F=None, S=None, sig=None, start=None):
    """mode：MODES 鍵或 dict；sig：訊號快取名稱（預設＝mode）；start：回測起點（預設 2005-01-01）。"""
    from panel import load as load_panel
    from feat import load_feat
    from signals import MODES, load_sig
    md = MODES[mode] if isinstance(mode, str) else mode
    P = P if P is not None else load_panel()
    F = F if F is not None else load_feat('AF_C' if md['bias2'] else 'AF')
    S = S if S is not None else load_sig(sig or mode)
    dates = P['dates']; T_, N_ = P['C'].shape; M = P['meta']
    CA = np.asarray(F['CA'], np.float32); OA = np.asarray(F['OA'], np.float32); VALID = np.asarray(F['VALID'])
    Cf = pd.DataFrame(CA).ffill().values
    d = S['daily']
    ctx = dict(dates=dates, T=T_, N=N_, codes=M['security_key'].to_numpy(), fix=not md['bias3'],
               Ov=OA, CA=CA, Cf=Cf, VALID=VALID,
               HA=np.asarray(F['HA'], np.float32), LA=np.asarray(F['LA'], np.float32),   # L-130：觸價單用
               RKX=np.asarray(S['RKX'], np.float32), RKV=np.asarray(S['RKV'], np.float32), ATRP=np.asarray(S['ATRP'], np.float32),
               below=d['below'].to_numpy(bool) | (d['xgate'].to_numpy(bool) if 'xgate' in d else False),   # L-117：X1 開倉閘門可併入 xgate
               MD=d['bear_mode'].to_numpy(bool), OFF=d['off'].to_numpy(bool),
               STALE=pd.DataFrame(VALID).rolling(20, min_periods=20).sum().values == 0,
               s0=int(dates.searchsorted(pd.Timestamp(start) if start else START)))
    if md['bias3']:
        # 參考實作的漲跌停鎖死推算（第 103–106 行），用未還原價
        RC = np.asarray(P['C'], float); RO = np.asarray(P['O'], float)
        RH = np.fmax(np.fmax(np.asarray(P['H'], float), RO), RC); RL = np.fmin(np.fmin(np.asarray(P['L'], float), RO), RC)
        lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10) - 0.005
        RCf = pd.DataFrame(RC).ffill().shift(1).values
        with np.errstate(invalid='ignore', divide='ignore'):
            r1 = RC / RCf - 1
        one = (RH == RL) & np.isfinite(RC)
        ctx['LD'] = one & (r1 <= -lim[:, None]); ctx['LU'] = one & (r1 >= lim[:, None])
    else:
        ctx['LD'] = np.asarray(P['LDL']); ctx['LU'] = np.asarray(P['LUL']); ctx['OSUS'] = np.asarray(P['OSUS'])
        # 最後有效日：之後不再於上市櫃交易（資料最後一日仍有效者不算）
        last = T_ - 1 - VALID[::-1].argmax(0); has = VALID.any(0)
        ctx['LAST'] = np.where(has & (last < T_ - 1), last, -1)
        dl = pd.to_datetime(M['delist_date'])
        rel = np.where(dl.notna(), dates.searchsorted(dl.fillna(dates[-1]).values), T_)
        ctx['REL'] = np.where(rel >= T_, T_, rel)       # 下市日（資料外＝永不回收）
    return ctx


def _sellable(c, t, j):
    if not c['fix']: return not c['LD'][t, j]
    return c['VALID'][t, j] and not c['LD'][t, j] and (np.isfinite(c['Ov'][t, j]) or c['OSUS'][t, j])


def _sellpx(c, t, j):
    if not c['fix']:
        o = c['Ov'][t, j]; return (o if not np.isnan(o) else c['Cf'][t, j]) * (1 - SLIP)
    return (c['CA'][t, j] if c['OSUS'][t, j] or not np.isfinite(c['Ov'][t, j]) else c['Ov'][t, j]) * (1 - SLIP)


def _buyable(c, t, j):
    px = c['Ov'][t, j]
    if np.isnan(px) or c['LU'][t, j]: return False
    if c['fix'] and (not c['VALID'][t, j] or c['OSUS'][t, j]): return False
    return True


_SZ = {}
def _size(path):
    if path not in _SZ: _SZ[path] = np.load(path, mmap_mode='r')
    return _SZ[path]


def _run(c, kind, off=0, seed=0, n=6, drop=0.2, cap=250, ban=None, delist='A', end=None, unlock=None, vstop=None, gate=None, size=None, xstop=None, pstop=None, ladder=None):
    """kind：'X'（X1）或 'V'（V4.1）。回傳（日報酬 Series、交易紀錄 list、持股比例 Series）。"""
    rng = np.random.default_rng(seed); dates = c['dates']; T_ = c['T'] if end is None else end
    s0 = c['s0'] + off; Ov, Cf, CA = c['Ov'], c['Cf'], c['CA']
    RK = c['RKX'] if kind == 'X' else c['RKV']
    cash = 1.0; pos = {}; sell, buy = {}, []; eq = np.full(c['T'], np.nan); ex = np.full(c['T'], np.nan); tr = []
    fix = c['fix']; bmode = delist[0] if isinstance(delist, tuple) else delist; disc = delist[1] if isinstance(delist, tuple) else 0.0

    def close(j, p, t, px, why):
        cash_in = p['sh'] * px * (1 - FEE - TAX)
        tr.append((c['codes'][j], dates[p['t0']], dates[t], p['own'] / p['E'], (cash_in + p.get('got', 0.0)) / p['own'] - 1, t - p['t0'], why, p['st'], p.get('lk', False)))
        return cash_in

    def val(p, j, t):
        return p['sh'] * (p['val'] if p.get('dead') else Cf[t, j])

    for t in range(s0, T_):
        if fix:                                                       # 凍結部位（B）在下市日回收
            for j in [j for j, p in pos.items() if p.get('dead') and t >= p['rel']]:
                p = pos.pop(j); cash += close(j, p, t, p['val'], '下市回收')   # 回收不計滑價，只扣手續費與稅
        keep = {}
        for j, why in sell.items():
            if j not in pos or pos[j].get('dead'): continue
            if not _sellable(c, t, j): keep[j] = why; continue
            p = pos.pop(j); cash += close(j, p, t, _sellpx(c, t, j), why)
        if buy:
            E = cash + sum(val(p, j, t - 1) for j, p in pos.items())
            for j in buy:
                if j in pos or len(pos) >= n: continue
                if not _buyable(c, t, j): continue
                sz = 1.0
                if size is not None and kind == 'V':                     # L-122：V4.1 部位大小倍數（訊號日 t−1）
                    sz = float(_size(size)[t - 1, j]); sz = 1.0 if not np.isfinite(sz) else sz
                px = Ov[t, j] * (1 + SLIP); own = min(E / n * sz, cash)
                if own <= 0: break
                p = dict(sh=own * (1 - FEE) / px, t0=t, px=px, own=own, E=E, st=False)
                if kind == 'X':
                    d = c['ATRP'][t - 1, j] * 4; p['d'] = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15); p['hi'] = px
                    if pstop is not None:                                 # L-129：路徑停損（N 日內收盤未達進場 × (1 + k × ATR%) → 出場）
                        a = c['ATRP'][t - 1, j]; p['tgt'] = px * (1 + pstop[1] * (a if np.isfinite(a) else 0.02)); p['ok'] = False
                    if xstop is not None and (xstop[2] is None or bool(_size(xstop[2])[t - 1, j])):   # L-125：替代出場（'hard'／'trail'，寬度 w）
                        p['xm'] = xstop[0]; p['d'] = xstop[1]
                else:
                    p['bear'] = bool(c['MD'][t - 1]); p['hold'] = int(rng.integers(18, 23))
                    if isinstance(vstop, tuple):                         # L-119：空頭停損寬度依 ATR（k × ATR%，限 lo–hi）
                        dv = c['ATRP'][t - 1, j] * vstop[0]; p['vd'] = vstop[1] if not np.isfinite(dv) else min(max(dv, vstop[1]), vstop[2])
                    else: p['vd'] = 0.05
                    if ladder is not None:                               # L-130：V4.1 分批停利＋ATR 硬停損（觸價單）
                        a = c['ATRP'][t - 1, j]; a = 0.02 if not np.isfinite(a) else float(a); b0 = float(Ov[t, j])
                        p['L'] = dict(base=b0, atr=a * b0, stage=0, stop=b0 - ladder['stop'] * a * b0)
                pos[j] = p; cash -= own
        sell, buy = keep, []
        if ladder is not None and kind == 'V':                        # L-130：盤中觸價（同日先停損、後停利；跳空以開盤價成交）
            sl = ladder.get('slip', SLIP)
            for j in [j for j, p in pos.items() if 'L' in p and not p.get('dead')]:
                if fix and not c['VALID'][t, j]: continue
                p = pos[j]; L_ = p['L']; o, h, l = float(Ov[t, j]), float(c['HA'][t, j]), float(c['LA'][t, j])
                if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l)): continue
                lk = c['LD'][t, j]
                if not lk and (o <= L_['stop'] or l <= L_['stop']):
                    px = min(o, L_['stop']) * (1 - sl)
                    pos.pop(j); cash += close(j, p, t, px, ('停損' if L_['stage'] == 0 else '保本') + f"@{L_['stage']}"); sell.pop(j, None); continue
                done = False
                while L_['stage'] < len(ladder['steps']):
                    k, frac, nstop = ladder['steps'][L_['stage']]; lvl = L_['base'] + k * L_['atr']
                    if h < lvl: break
                    px = max(o, lvl) * (1 - sl); L_['stage'] += 1
                    if frac >= 1.0:
                        pos.pop(j); cash += close(j, p, t, px, f"停利@{L_['stage']}"); sell.pop(j, None); done = True; break
                    sh = p['sh'] * frac; got = sh * px * (1 - FEE - TAX)
                    p['sh'] -= sh; p['got'] = p.get('got', 0.0) + got; cash += got
                    L_['stop'] = L_['base'] + nstop * L_['atr']
                if done: continue
        if fix:                                                       # 最後有效日：下市出清（A）或凍結（B）
            for j in [j for j in pos if c['LAST'][j] == t and not pos[j].get('dead')]:
                p = pos[j]; lk = bool(c['LD'][t, j]); p['lk'] = lk
                if bmode == 'B' and lk:
                    p['dead'] = True; p['val'] = CA[t, j] * (1 - disc); p['rel'] = c['REL'][j]
                else:
                    pos.pop(j); cash += close(j, p, t, CA[t, j] * (1 - SLIP), '下市出清'); sell.pop(j, None)
        mv_ = sum(val(p, j, t) for j, p in pos.items()); eq[t] = cash + mv_; ex[t] = mv_ / eq[t]
        for j, p in pos.items():
            if p.get('dead'): continue
            if c['STALE'][t, j]: p['st'] = True
            if fix and not c['VALID'][t, j]:
                if kind == 'X':
                    if t - p['t0'] + 1 >= cap: sell[j] = '250日上限'
                else:
                    if c['OFF'][t] and t - p['t0'] >= 1: sell[j] = 'IAKF'
                    elif t - p['t0'] + 1 >= p['hold']: sell[j] = '持有期滿'
                continue
            cc = Cf[t, j] if not fix else CA[t, j]
            if np.isnan(cc): continue
            if kind == 'X':
                p['hi'] = max(p['hi'], cc)
                if p.get('xm') == 'hard':
                    if cc < p['px'] * (1 - p['d']): sell[j] = '硬停損'
                    elif t - p['t0'] + 1 >= cap: sell[j] = '250日上限'
                    continue
                if p.get('xm') == 'trail':
                    if cc < p['hi'] * (1 - p['d']): sell[j] = '移動停損'
                    elif t - p['t0'] + 1 >= cap: sell[j] = '250日上限'
                    continue
                if 'tgt' in p and cc >= p['tgt']: p['ok'] = True
                if cc < p['hi'] * (1 - p['d']): sell[j] = '移動停損'
                elif p['hi'] >= p['px'] * 1.06 and cc < p['px'] * 1.005: sell[j] = '保本'
                elif 'tgt' in p and not p['ok'] and t - p['t0'] + 1 >= pstop[0]: sell[j] = '路徑停損'
                elif t - p['t0'] + 1 >= cap: sell[j] = '250日上限'
            elif 'L' in p:
                if c['OFF'][t] and t - p['t0'] >= 1: sell[j] = 'IAKF'
                elif ladder.get('keep_time', True) and t - p['t0'] + 1 >= p['hold']: sell[j] = '持有期滿'
                elif t - p['t0'] + 1 >= cap: sell[j] = '250日上限'
            else:
                if c['OFF'][t] and t - p['t0'] >= 1: sell[j] = 'IAKF'
                elif vstop != 'none' and (c['MD'][t] or p['bear']) and cc < p['px'] * (1 - p.get('vd', 0.05)): sell[j] = '空頭5%停損'
                elif t - p['t0'] + 1 >= p['hold']: sell[j] = '持有期滿'
        free = n - len(pos) + len(set(sell) & set(pos))
        if kind == 'X':
            g0 = c['below'][t] if gate is None else bool(gate[t])            # M-144：gate 可整條替換 X1 閘門
            gate_ = g0 and not (unlock is not None and unlock[t])            # M-109：unlock 窗口內解除 X1 閘門
        else: gate_ = c['OFF'][t]
        if free > 0 and not gate_:
            cnd = np.where(~np.isnan(RK[t]))[0]
            if ban is not None: cnd = cnd[~np.isin(cnd, ban)]
            if drop: cnd = cnd[rng.random(len(cnd)) >= drop]
            buy = [j for j in cnd[np.argsort(RK[t, cnd])] if j not in pos and j not in sell][:free]
    for j, p in pos.items():
        px = p['val'] if p.get('dead') else Cf[T_ - 1, j] * (1 - SLIP)
        close(j, p, T_ - 1, px, '期末未平倉')
    return (pd.Series(eq, index=dates).dropna().pct_change().fillna(0), tr, pd.Series(ex, index=dates).dropna())


def runX(c, off=0, seed=0, n=6, drop=0.2, cap=250, ban=None, delist='A', end=None, unlock=None, gate=None, xstop=None, pstop=None):
    return _run(c, 'X', off, seed, n, drop, cap, ban, delist, end, unlock, None, gate, None, xstop, pstop)


def runV(c, off=0, seed=0, n=4, drop=0.2, ban=None, delist='A', end=None, vstop=None, size=None, ladder=None):
    return _run(c, 'V', off, seed, n, drop, 250, ban, delist, end, None, vstop, None, size, None, None, ladder)


def mix(a, b, wa=0.6):
    """合併兩個子帳戶日報酬，每月第一個交易日再平衡回 wa : 1−wa（同參考實作）。"""
    a, b = a.align(b, fill_value=0); va, vb, out = wa, 1 - wa, np.empty(len(a))
    m = a.index.to_period('M'); av, bv = a.values, b.values
    for i in range(len(a)):
        if i > 0 and m[i] != m[i - 1]: tot = va + vb; va, vb = tot * wa, tot * (1 - wa)
        prev = va + vb; va *= 1 + av[i]; vb *= 1 + bv[i]; out[i] = (va + vb) / prev - 1
    return pd.Series(out, index=a.index)


# ---------- 多路徑（多程序）
_CTX = None
def _init(mode, sig, start):
    global _CTX
    _CTX = make_ctx(mode, sig=sig, start=start)

def _one(args):
    i, kind, kw = args
    f = runX if kind == 'X' else runV
    r, tr, ex = f(_CTX, off=i % 20, seed=1000 + i, **kw)
    return i, kind, r, tr, ex


def run_paths(mode='compat', npath=100, x_kw=None, v_kw=None, workers=None, sig=None, start=None):
    """回傳 dict：X、V 為各路徑（日報酬、交易、曝險）list。路徑 i：起始日錯開 i mod 20、種子 1000 + i、每日剔除 20% 候選（L-113 1b）。"""
    from multiprocessing import get_context
    x_kw = x_kw or {}; v_kw = v_kw or {}
    jobs = [(i, 'X', x_kw) for i in range(npath)] + [(i, 'V', v_kw) for i in range(npath)]
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    out = {'X': [None] * npath, 'V': [None] * npath}
    with get_context('spawn').Pool(workers, initializer=_init, initargs=(mode, sig, start)) as pool:
        for i, kind, r, tr, ex in pool.imap_unordered(_one, jobs):
            out[kind][i] = (r, tr, ex)
    return out
