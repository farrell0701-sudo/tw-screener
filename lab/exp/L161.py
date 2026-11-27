# L-161：定版關掉既有大盤防禦，改用多空點縮倉。判準見 registry_L.md L-161。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NP = 100; SG = ['2005–15', '2016–20', '2021–26']


def bear_b(ixp, dates, b):
    T = len(ixp); m_ = dates.to_period('M'); first = pd.Series(np.arange(T), index=dates).groupby(m_).min(); mi = {p: i for i, p in enumerate(first.index)}
    A = pd.Series(np.abs(np.diff(ixp, prepend=np.nan))).rolling(14).mean().to_numpy(); st = np.zeros(T, bool); cur = False
    for t in range(T):
        k = mi[m_[t]]; a = first.iloc[k - 1] if k > 0 else first.iloc[0]; w = ixp[a:t + 1]; w = w[np.isfinite(w)]
        if len(w) == 0 or not np.isfinite(A[t]) or not np.isfinite(ixp[t]): st[t] = cur; continue
        ln = (w.max() + w.min()) / 2
        if not cur and ixp[t] < ln - b * A[t]: cur = True
        elif cur and ixp[t] > ln: cur = False
        st[t] = cur
    return st


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T = len(dates)
    signals.build('fix', mode='fix', P=P, F=F)
    signals.build('L161_off', mode='fix', P=P, F=F, v_iakf=False, v_vol=False)
    ixp = P['index']['TAIEX_P'].reindex(dates).ffill().to_numpy()
    B = sim.run_paths('fix', NP, sig='fix'); BM = [sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(NP)]
    O = sim.run_paths('fix', NP, sig='L161_off', x_kw={'gate': np.zeros(T, bool)}, v_kw={'vstop': 'none'})
    OM = [sim.mix(O['X'][i][0], O['V'][i][0]) for i in range(NP)]
    bt = E.seg_table(BM); bpm = E.path_mean(BM)
    def overlay(r, st):
        st = pd.Series(st, index=dates).reindex(r.index).fillna(False).to_numpy()
        ex = np.where(np.r_[False, st[:-1]], 0.2, 1.0); cost = np.r_[0, np.abs(np.diff(ex))] * 0.00385
        return pd.Series(ex * r.to_numpy() - cost, index=r.index)
    rows = {'新基準（防禦全開）': (bt, None, BM), 'D0 防禦全關、無疊加': (E.seg_table(OM), E.boot_p(E.path_mean(OM), bpm), OM)}
    for b in (0.0, 0.5, 1.0, 1.5):
        st = bear_b(ixp, dates, b); Ov = [overlay(r, st) for r in OM]
        rows[f"全關＋多空點 b={b}{'（主）' if b == 1.0 else ''}"] = (E.seg_table(Ov), E.boot_p(E.path_mean(Ov), bpm), Ov)
    L = ['## 定版：既有防禦全關 vs 多空點替代（Calmar（年化）；100 路徑中位數）\n', '| 版本 | ' + ' | '.join(SG) + ' | 2005–15 回撤 | 2016–26 回撤 | 全期回撤 | P（16–26） |', '|---|---|---|---|---|---|---|---|']
    for nm, (t, p, _) in rows.items():
        L.append(f'| {nm} | ' + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SG) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {t.loc['全期', '回撤']:.1%} | {'' if p is None else f'{p:.2f}'} |")
    # 逐年
    yb = lambda M: E.path_mean(M).pipe(lambda m: m.groupby(m.index.year).apply(lambda x: (1 + x).prod() - 1))
    Y = {nm: yb(v[2]) for nm, v in rows.items()}; ys = [y for y in Y['新基準（防禦全開）'].index if y >= 2005]
    L += ['\n## 逐年報酬（路徑平均）\n', '| 版本 | ' + ' | '.join(str(y) for y in ys) + ' |', '|' + '---|' * (len(ys) + 1)]
    for nm, y in Y.items(): L.append(f'| {nm[:14]} | ' + ' | '.join(f'{y.get(k, np.nan):+.0%}' for k in ys) + ' |')
    d0 = rows['D0 防禦全關、無疊加'][0]
    rm = all(d0.loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] - 0.05 for s in ('2016–20', '2021–26')) and d0.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    tm, pm = rows['全關＋多空點 b=1.0（主）'][:2]
    c1 = all(tm.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')); c2 = pm >= 0.8; c3 = tm.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    c4 = any(all(rows[f'全關＋多空點 b={x}'][0].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for x in (0.5, 1.5))
    veto = all(rows[k][0].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in rows if k.startswith('全關＋'))
    ok = c1 and c2 and c3 and c4 and not veto
    L.append(f"\n## 判準\n\n- D0（拿掉防禦，L-106）：{'可拿掉' if rm else '不可拿掉'}\n- 主版本（全關＋多空點 b=1.0）：兩段上升 {'✓' if c1 else '✗'}；P {pm:.2f} {'✓' if c2 else '✗'}；05–15 回撤 {'✓' if c3 else '✗'}；高原 {'✓' if c4 else '✗'}；否決 {'觸發' if veto else '未觸發'} → **{'通過' if ok else '不通過'}**")
    head = f"L-161 定版防禦全關：D0 {'可拿掉' if rm else '不可拿掉'}；改用多空點 b=1.0 {'通過' if ok else '不通過'}（P {pm:.2f}）"
    return dict(summary='# L-161 結果\n\n**' + head + '**\n\n' + '\n'.join(L), headline=head, n_variants=5)
