# M-124：低淨值比＋獲利穩定在崩盤／熊市的表現（描述性、前視事件日）。定義見 registry_M.md M-124。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from chips import _staged, to_grid
from M108 import episodes
import M120

REB = 60


def eps_table():
    fs = []
    for y in range(2003, 2027):
        f = M120.TWD / 'derived' / 'financials' / f'{y}.parquet'
        if f.exists():
            d = pd.read_parquet(f); fs.append(d[(d.statement == 'income') & (d.type == 'EPS')][['stock_id', 'period_end', 'value', 'available_date']])
    return pd.concat(fs).drop_duplicates(['stock_id', 'period_end'], keep='last').sort_values(['stock_id', 'period_end'])


def stable(E, t):
    sub = E[E.available_date < t]
    out = {}
    for s, g in sub.groupby('stock_id'):
        g = g.tail(4)
        if len(g) == 4 and (t - g.period_end.iloc[-1]).days <= 210:
            out[s] = bool((g.value > 0).all())
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    PB = to_grid(_staged('TaiwanStockPER', ['date', 'stock_id', 'PBR']), 'PBR', P); PB[~(PB > 0)] = np.nan
    PB = pd.DataFrame(PB).ffill(limit=5).to_numpy()
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    sid = P['meta']['stock_id'].astype(str).to_numpy()
    E = eps_table()
    ix = P['index']['TAIEX_P'].reindex(dates).ffill()
    EP = [e for e in episodes(ix.loc['2005':]) if e[0] >= pd.Timestamp('2005-10-01')]
    L = ['# M-124 結果：低淨值比＋獲利穩定在崩盤／熊市的表現（描述性、前視事件日、所有代碼）\n',
         '高點前一交易日分組，高點收盤等權買進持有；報酬／期間最大回落；低 PB＝層內 PBR 最低 20%、高 PB＝最高 20%；獲利穩定＝最近 4 季 EPS 皆 > 0。\n']
    agg = []
    for k, (pk, tr, depth) in enumerate(EP, 1):
        a, b = dates.get_loc(pk), dates.get_loc(tr); c = min(b + REB, T - 1); s = a - 1
        stb = stable(E, pk)
        st = np.array([stb.get(x, False) for x in sid])
        L.append(f"## 第 {k} 次：{pk.date()} → {tr.date()}（指數 {depth:.1%}；之後 {REB} 日指數 {ix.iloc[c] / ix.iloc[b] - 1:+.1%}）\n")
        L += ['| 層 | 組 | 檔數 | 下跌段報酬 | 下跌段最大回落 | 反彈 60 日（續抱） |', '|---|---|---|---|---|---|']
        for layer, mask in (('全體', U[s]), ('T_ld 前 10%', pool[s])):
            pb = np.where(mask, PB[s], np.nan); has = mask & np.isfinite(pb)
            lo_, hi_ = np.nanpercentile(pb[has], [20, 80]) if has.sum() >= 10 else (np.nan, np.nan)
            groups = {'層全體': mask, '有淨值比者': has, '獲利穩定': mask & st, '低 PB': has & (pb <= lo_), '低 PB＋獲利穩定': has & (pb <= lo_) & st, '高 PB': has & (pb >= hi_)}
            for gn, gm in groups.items():
                jj = np.nonzero(gm)[0]
                if len(jj) == 0: L.append(f'| {layer} | {gn} | 0 | | | |'); continue
                with np.errstate(all='ignore'):
                    v = np.nanmean(Cf[a:c + 1, jj] / Cf[a, jj], axis=1)
                dn = v[b - a] - 1; dd = (v[:b - a + 1] / np.maximum.accumulate(v[:b - a + 1]) - 1).min(); rb = v[c - a] / v[b - a] - 1
                L.append(f'| {layer} | {gn} | {len(jj)} | {dn:+.1%} | {dd:.1%} | {rb:+.1%} |')
                agg.append(dict(k=k, layer=layer, g=gn, n=len(jj), dn=dn, dd=dd, rb=rb))
        L.append('')
    A = pd.DataFrame(agg)
    L += ['## 彙總（各次平均；勝過同層全體的次數）\n', '| 層 | 組 | 平均檔數 | 下跌段平均 | 最大回落平均 | 反彈 60 日平均 | 下跌段勝過層全體 | 反彈勝過層全體 |', '|---|---|---|---|---|---|---|---|']
    for layer in ('全體', 'T_ld 前 10%'):
        base = A[(A.layer == layer) & (A.g == '層全體')].set_index('k')
        for gn in ('層全體', '獲利穩定', '低 PB', '低 PB＋獲利穩定', '高 PB'):
            x = A[(A.layer == layer) & (A.g == gn)].set_index('k')
            if x.empty: continue
            w1 = (x.dn > base.dn.reindex(x.index)).sum(); w2 = (x.rb > base.rb.reindex(x.index)).sum()
            L.append(f"| {layer} | {gn} | {x.n.mean():.0f} | {x.dn.mean():+.1%} | {x.dd.mean():.1%} | {x.rb.mean():+.1%} | {w1}/{len(x)} | {w2}/{len(x)} |")
    return dict(summary='\n'.join(L), headline=f'{len(EP)} 次熊市描述完成', n_variants=0)
