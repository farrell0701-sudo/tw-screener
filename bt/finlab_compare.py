# -*- coding: utf-8 -*-
"""
finlab 資料上的版本比較：V3C-PRO vs 300 萬版，並各自跑同日候選隨機挑選的分布。
需要與 finlab_v3c_pro.py 放在同一個資料夾。
執行時間：建訊號每版約 30 秒，每次模擬約 1～3 秒；預設每組隨機 100 次，全部約 10～20 分鐘。
"""
import time
import numpy as np
import pandas as pd
import finlab_v3c_pro as F

N_RANDOM = 100          # 每組隨機挑選次數（越多越準、越久）
TRADE_START = '2016-01-01'

# ───────── 300 萬版（v3c）專用指標 ─────────
P300 = dict(F.P, dist_max=0.20, slip_flat=0.001, min_score=3,
            vol_look=5, vol_max=3.0, s_rng=16.49, s_ema=11.9, s_chg=0.0268, s_cap=10)


def features_300(adj, ah, al, raw, vol, cap_yi, p=P300):
    C = adj; n = np.arange(len(C))
    ema = C.ewm(span=p['trend_len'], adjust=False).mean()
    slope = (ema / ema.shift(p['slope_look']) - 1) * 100 / p['slope_look']
    brk = C > C.shift(1).rolling(p['brk_len']).max()
    vr = vol / vol.rolling(20).mean()
    novol = vr.shift(1).rolling(p['vol_look']).max() <= p['vol_max']
    dist = C / ah.rolling(p['hh_len']).max() - 1
    pc = C.shift(1)
    tr = pd.concat([ah - al, (ah - pc).abs(), (al - pc).abs()], axis=1).max(axis=1)
    atrp = tr.ewm(alpha=1 / p['atr_len'], adjust=False).mean() / C * 100
    chg = C / pc - 1
    adv = (raw * vol).rolling(p['adv_len']).mean()
    rng20 = (ah.rolling(20).max() - al.rolling(20).min()) / C * 100
    d60 = (C / C.ewm(span=60, adjust=False).mean() - 1) * 100
    score = ((atrp <= p['s_atr']).astype(int) + (dist >= p['s_dist']).astype(int) + (rng20 <= p['s_rng']).astype(int)
             + (d60 <= p['s_ema']).astype(int) + (chg <= p['s_chg']).astype(int) + int(cap_yi >= p['s_cap']))
    pos_main = dist >= p['pos_max']
    rescue = (~pos_main) & (atrp <= p['rescue_atr'])
    sig = ((n >= p['hh_len']) & (slope >= p['slope_min']) & brk & novol & (pos_main | rescue)
           & (chg < p['limit_up']) & (adv >= p['adv_min']) & (score >= p['min_score']))
    return pd.DataFrame({'sig': sig, 'score': score, 'atrp': atrp, 'adv': adv, 'rescue': rescue})


def build_300(raw, adj, hi, lo, vol, mkt_ok, ind_map, cap_map, p=P300):
    f = adj / raw; ah, al = hi * f, lo * f
    out = {k: {} for k in ('sig', 'score', 'atrp', 'adv', 'rescue')}
    for c in raw.columns:
        m = adj[c].notna() & raw[c].notna() & vol[c].fillna(0).gt(0)
        if m.sum() < p['hh_len'] + 1:
            continue
        d = features_300(adj[c][m], ah[c][m], al[c][m], raw[c][m], vol[c][m], cap_map.get(c, 0), p)
        for k in out:
            out[k][c] = d[k]
    W = {k: pd.DataFrame(v).reindex(raw.index) for k, v in out.items()}
    sig = W['sig'].fillna(False).astype(bool)
    sig = sig.mul(mkt_ok.reindex(raw.index).ffill().fillna(False).astype(bool), axis=0)
    sig[[c for c in sig.columns if ind_map.get(c) in p['excl_ind']]] = False
    return sig, W['score'], W['atrp'], W['adv'], W['rescue']


def capital_map(data, cols):
    """股本（億元），用最新一期財報；自動判斷單位。"""
    try:
        s = data.get('financial_statement:股本').ffill().iloc[-1]
        unit = 1e8 if s.median() > 1e8 else 1e5          # 元 或 千元
        cap = (s / unit).to_dict()
        print('股本單位', '元' if unit == 1e8 else '千元', '｜台積電股本(億)', round(cap.get('2330', float('nan')), 1))
        return cap
    except Exception as e:  # noqa: BLE001
        print('股本抓不到，股本項一律不給分：', e)
        return {}


def run_set(label, built, p, raw, adj, ix, rows):
    sig, score, atrp, adv, rescue = built
    t0 = time.time()
    E, T, W, N = F.simulate(sig, score, atrp, adv, rescue, raw, adj, dict(p, trade_start=TRADE_START))
    m = F.metrics(E, T, ix)
    rows.append(dict(版本=label, 排序='分數→低價', 年化=m['年化'], 最大回撤=m['最大回撤'], Sharpe=m['Sharpe'],
                     交易數=m.get('交易數', 0), 勝率=m.get('勝率', 0)))
    print(f'{label} 基準 年化 {m["年化"]:.1%} 回撤 {m["最大回撤"]:.1%}（{time.time() - t0:.0f} 秒）', flush=True)
    print('   逐年', m['逐年'], flush=True)
    for rk in ('score_random', 'random'):
        res = []
        for s in range(N_RANDOM):
            E2, T2, _, _ = F.simulate(sig, score, atrp, adv, rescue, raw, adj, dict(p, trade_start=TRADE_START, rank=rk, seed=s))
            mm = F.metrics(E2, T2)
            res.append((mm['年化'], mm['最大回撤'], mm['Sharpe']))
        a = np.array(res)
        pct = (a[:, 0] < m['年化']).mean()
        rows.append(dict(版本=label, 排序='分數→隨機' if rk == 'score_random' else '完全隨機',
                         年化=np.median(a[:, 0]), 年化5=np.percentile(a[:, 0], 5), 年化95=np.percentile(a[:, 0], 95),
                         最大回撤=np.median(a[:, 1]), Sharpe=np.median(a[:, 2]), 基準百分位=pct))
        print(f'   {rk} ×{N_RANDOM} 年化中位數 {np.median(a[:, 0]):.1%}（5%～95%：{np.percentile(a[:, 0], 5):.1%}～'
              f'{np.percentile(a[:, 0], 95):.1%}） 回撤中位數 {np.median(a[:, 1]):.1%} ｜ 基準排在第 {pct:.0%} 百分位', flush=True)


if __name__ == '__main__':
    from finlab import data
    S0 = F.P['data_start']
    raw = data.get('price:收盤價').loc[S0:]; adj = data.get('etl:adj_close').loc[S0:]
    hi = data.get('price:最高價').loc[S0:]; lo = data.get('price:最低價').loc[S0:]
    vol = data.get('price:成交股數').loc[S0:]; op = data.get('price:開盤價').loc[S0:]
    sc = data.get('security_categories')
    sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
    bad = set(sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str))
    cols = [c for c in raw.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('00')
            and str(c) not in bad and all(c in x.columns for x in (adj, hi, lo, vol))]
    raw, adj, hi, lo, vol, op = raw[cols], adj[cols], hi[cols], lo[cols], vol[cols], op.reindex(columns=cols)
    hi = np.fmax(np.fmax(hi, op.reindex_like(hi)), raw); lo = np.fmin(np.fmin(lo, op.reindex_like(lo)), raw)
    ix = data.get('stock_index_price:收盤指數').loc[S0:]['上市發行量加權股價指數']
    e60 = ix.ewm(span=60, adjust=False).mean(); mkt_ok = (e60 / e60.shift(20) - 1) > 0
    ind_map = data.get('director_compensation:產業類別').ffill().iloc[-1].to_dict()
    cap_map = capital_map(data, cols)

    rows = []
    t0 = time.time(); b_pro = F.build_signals(raw, adj, hi, lo, vol, mkt_ok, ind_map); print(f'PRO 訊號 {time.time() - t0:.0f} 秒')
    run_set('V3C-PRO', b_pro, F.P, raw, adj, ix, rows); del b_pro
    t0 = time.time(); b_300 = build_300(raw, adj, hi, lo, vol, mkt_ok, ind_map, cap_map); print(f'300 萬版訊號 {time.time() - t0:.0f} 秒')
    run_set('300 萬版', b_300, P300, raw, adj, ix, rows)

    R = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(R.round(4).to_string(index=False))
    R.to_csv('finlab_compare.csv', index=False, encoding='utf-8-sig')
