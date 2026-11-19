# L-111：空頭期指數期貨避險疊加（2026-11-16 事前登記）。判準見 registry_L.md L-111（L-109 空頭避險型）。
# 每條路徑：合併版日報酬 R、持股比例 E（子帳戶曝險依當日資金權重加權）；
# 第 t 日價格指數收盤 < MA → 第 t+1 日起持有空單；空單損益＝−報酬指數日報酬 × 空單比例；
# 成本：空單名目變動 0.05%（進出場與調整），啟動期間每月第一個交易日轉倉 0.05%。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100; COST = 0.0005
VARS = {'H1': (200, 'E', 1.0), 'H2': (200, 'E', 0.5), 'H3': (200, 'F', 0.5), 'H1_MA150': (150, 'E', 1.0), 'H1_MA250': (250, 'E', 1.0)}
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def mix_with_expo(a, b, ea, eb, wa=0.6):
    """同 sim.mix 的每月再平衡；另回傳合併持股比例（以當日收盤後的子帳戶資金權重加權）。"""
    a, b = a.align(b, fill_value=0); ea = ea.reindex(a.index).fillna(0); eb = eb.reindex(a.index).fillna(0)
    va, vb = wa, 1 - wa; out = np.empty(len(a)); ex = np.empty(len(a))
    m = a.index.to_period('M'); av, bv, eav, ebv = a.values, b.values, ea.values, eb.values
    for i in range(len(a)):
        if i > 0 and m[i] != m[i - 1]: tot = va + vb; va, vb = tot * wa, tot * (1 - wa)
        prev = va + vb; va *= 1 + av[i]; vb *= 1 + bv[i]; out[i] = (va + vb) / prev - 1
        ex[i] = (va * eav[i] + vb * ebv[i]) / (va + vb)
    return pd.Series(out, index=a.index), pd.Series(ex, index=a.index)


def main():
    P = load_panel(); F = load_feat('AF')
    signals.build('fix', mode='fix', P=P, F=F)
    t0 = time.time(); R = sim.run_paths('fix', NPATH, sig='fix'); log(f'（基準 100 路徑：{time.time() - t0:.0f} 秒）')
    base = [mix_with_expo(R['X'][i][0], R['V'][i][0], R['X'][i][2], R['V'][i][2]) for i in range(NPATH)]
    idx = base[0][0].index
    for r, _ in base[1:]: idx = idx.union(r.index)
    base = [(r.reindex(idx).fillna(0), e.reindex(idx).fillna(0)) for r, e in base]
    px = P['index']['TAIEX_P'].reindex(idx).ffill(); tr = P['index']['TAIEX_TR'].reindex(idx).ffill(); trr = tr.pct_change().fillna(0)
    month_first = pd.Series(idx.to_period('M'), index=idx); month_first = (month_first != month_first.shift(1)).values
    out = {}; info = {}
    for nm, (ma, kind, k) in VARS.items():
        act = (px < px.rolling(ma).mean()).shift(1, fill_value=False).values        # t 日收盤訊號 → t+1 日起持有
        paths = []; hp_all = []
        for r, e in base:
            h = np.where(act, (k * e.shift(1).fillna(0).values) if kind == 'E' else k, 0.0)
            pnl = -trr.values * h
            cost = COST * np.abs(np.diff(np.r_[0.0, h])) + COST * h * (month_first & act)
            paths.append(pd.Series(r.values + pnl - cost, index=idx)); hp_all.append(pnl)
        out[nm] = paths
        info[nm] = dict(act=pd.Series(act, index=idx), hp=np.mean(hp_all, axis=0))
    B = [r for r, _ in base]
    bt = E.seg_table(B); bpm = E.path_mean(B)
    log('## 合併版（年化 / 回撤 / Calmar；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | P（16-26） |\n|---|---|---|---|---|---|---|')
    rows = [('新基準', bt, '')]
    tabs = {}
    for nm, ps in out.items():
        t = E.seg_table(ps); tabs[nm] = t
        rows.append((nm, t, f"{E.boot_p(E.path_mean(ps), bpm):.2f}"))
    for nm, t, p in rows:
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, '年化']:.1%} / {t.loc[s, '回撤']:.1%} / {t.loc[s, 'Calmar']:.2f}" for s in SEGS) + f" | {p} |")
    # 必報
    log('\n## 避險啟動與損益（H1）\n')
    a = info['H1']['act']; hp = pd.Series(info['H1']['hp'], index=idx)
    rm = E.path_mean(B)
    for s in ('2005–15', '2016–20', '2021–26'):
        lo, hi = E.SEGS[s]; m = (idx >= lo) & (idx <= hi)
        aa = a.values & m
        corr = np.corrcoef(rm.values[aa], hp.values[aa])[0, 1] if aa.sum() > 10 else np.nan
        log(f"- {s}：啟動天數 {aa.sum() / m.sum():.0%}；啟動期間合併報酬與空單損益相關 {corr:+.2f}；空單損益合計（對數近似）{hp.values[m].sum():+.1%}")
    up20 = px.pct_change(20) > 0.10
    reb = a.values & up20.values
    log(f"- 空頭中反彈段（指數 20 日漲 > 10% 且避險啟動）：{reb.sum()} 日，空單損益合計 {hp.values[reb].sum():+.1%}")
    # 判準
    h1 = tabs['H1']
    dd_imp = h1.loc['2005–15', '回撤'] - bt.loc['2005–15', '回撤']
    ann_drop = bt.loc['全期', '年化'] - h1.loc['全期', '年化']
    c1 = dd_imp >= 0.03; c2 = ann_drop <= 0.02
    plateau_fail = any(tabs[v].loc['2005–15', '回撤'] < bt.loc['2005–15', '回撤'] for v in ('H1_MA150', 'H1_MA250'))
    ok = c1 and c2 and not plateau_fail
    log('\n## 判準（事前固定，L-111／L-109 空頭避險型）\n')
    log(f"- H1 2005–15 回撤改善 {dd_imp * 100:+.1f}pt（需 ≥ +3pt）：{'成立' if c1 else '**不成立**'}")
    log(f"- H1 全期年化下降 {ann_drop * 100:.1f}pt（需 ≤ 2pt）：{'成立' if c2 else '**不成立**'}")
    log(f"- 高原（MA150、MA250 不得使 05-15 回撤變差）：{'**失敗**' if plateau_fail else '成立'}")
    others = [v for v in ('H2', 'H3') if (tabs[v].loc['2005–15', '回撤'] - bt.loc['2005–15', '回撤'] >= 0.03 and bt.loc['全期', '年化'] - tabs[v].loc['全期', '年化'] <= 0.02)]
    log(f"- 其他變體單獨達標：{'、'.join(others) if others else '無'}（H1 未通過時，單一變體通過不採用）")
    log(f"\n**結論：{'通過（建議採用，待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-111 避險疊加：{'通過' if ok else '不採用'}（H1 05-15 回撤 {dd_imp * 100:+.1f}pt、全期年化 −{ann_drop * 100:.1f}pt）"
    return dict(summary='# L-111 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v) for k, v in out.items()}})
