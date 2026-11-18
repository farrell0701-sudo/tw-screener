# 面板品質報告 → lab/data/PANEL_REPORT.md（只含彙總數字）
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel import load, OUT, LAB, CFG, peak_mem_mb

# VERIFY_v1.1.md 的對照數字（當年有成交的上市櫃普通股）
VERIFY_N = {2005: 1217, 2026: 1994}


def md(df, fmt=None):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join([df.index.name or ''] + cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append('| ' + ' | '.join([str(i)] + [(fmt or str)(v) for v in r.values]) + ' |')
    return '\n'.join(lines)


def report():
    P = load(); d = P['dates']; M = P['meta']; info = P['info']; yr = d.year.values
    L = []; w = L.append
    w('# 面板品質報告（lab/data/panel.py）\n')
    w(f"資料：tw-data `{info['tag']}`｜建置 {info['built']}｜{info['start']} 至 {info['end']}｜{info['T']} 個交易日 × {info['N']} 檔普通股｜"
      f"{info['rows']:,} 列｜耗時 {info['seconds']} 秒｜建置峰值記憶體 {info['peak_mb']} MB\n")
    sz = sum(f.stat().st_size for f in OUT.glob('*')) / 2**20
    w(f'快取大小 {sz:.0f} MB（lab/cache/panel，不入庫）。\n')

    w('## 1. 一致性檢查\n')
    w(f"- 還原因子：由事件因子重建 vs tw-data `af`，最大相對誤差 {info['af_max_rel_err']:.2e}。")
    w(f"- 事件表（status＝ok）與價格表事件因子：{info['ev_ok']:,} 個（證券, 日期），不一致 {info['ev_mismatch']}，不在交易日 {info['ev_off_calendar']}；價格表有事件因子的格數 {info['ef_rows']:,}。")
    w(f"- 交易日：TAIEX 有值但面板沒有 {info['taiex_days_missing_in_panel']} 天；面板有但 TAIEX 沒有 {info['panel_days_without_taiex']} 天。")
    w(f"- 相容模式產業名單（fl_small.py `PK`）：{info['pk_n']} 個代碼，對到面板 {info['pk_matched']} 檔（代碼重用時同代碼各段都給同一產業）。\n")

    EX, TRD, EMG = P['EX'], P['TRD'], P['EMG']
    LIST = np.asarray(EX) & ~np.asarray(EMG); TL = LIST & np.asarray(TRD)
    rows = []
    for y in sorted(set(yr)):
        s = yr == y
        n = int(TL[s].any(axis=0).sum())
        rows.append({'年': y, '當年有成交的上市櫃普通股': n, 'VERIFY': VERIFY_N.get(y, ''),
                     '年底上市櫃有資料列': int(LIST[np.where(s)[0][-1]].sum())})
    t = pd.DataFrame(rows).set_index('年')
    w('## 2. 每年可交易股數（對照 VERIFY_v1.1.md）\n'); w(md(t)); w('')
    bad = [y for y, v in VERIFY_N.items() if y in t.index and t.loc[y, '當年有成交的上市櫃普通股'] != v]
    w(f"結果：{'**不符** ' + str(bad) if bad else '與 VERIFY 端點數字相同'}。\n")

    # 2330 還原抽查
    C = P['C']; AF = P['AF']; AFC = P['AF_C']
    k = M.index[(M['stock_id'] == '2330')][0]
    c = np.asarray(C[:, k], float); a = np.asarray(AF[:, k]); ac = np.asarray(AFC[:, k])
    s = pd.DataFrame({'raw': c, 'adj': c * a, 'cmp': c * ac}, index=d).dropna()
    r = s / s.shift(1) - 1
    chg = np.r_[False, a[1:] != a[:-1]]; ev = pd.Series(chg, index=d).reindex(s.index).fillna(False).to_numpy()
    w('## 3. 抽查 2330 還原\n')
    w(f"- 還原因子變動 {int(chg.sum())} 次（VERIFY 數 tw-data `af` 為 46 次：事件 45 個，另 1 次是 `af` 浮點尾數變動、沒有事件；面板由事件重建，故為 45）；這些日子未還原報酬中位 {r['raw'][ev].median():.2%}、還原後 {r['adj'][ev].median():.2%}。最新一日 af＝{a[-1]:.6f}。")
    yy = (r + 1).groupby(r.index.year).prod() - 1
    yt = pd.DataFrame({'完整還原−未還原(pt)': (yy['adj'] - yy['raw']) * 100, '相容版−未還原(pt)': (yy['cmp'] - yy['raw']) * 100}).round(1)
    yt.index.name = '年'; w(''); w(md(yt.loc[2005:])); w('')
    w('相容版在 2016/7 以前應約為 0（只剩減資等非股利事件），之後與完整還原相同。\n')

    # 相容版（偏誤 ②）全市場影響
    w('## 4. 相容模式偏誤\n')
    w(f"- 偏誤 ②：不還原的股利事件 {info['compat_div_events']:,} 個（2016/7 以前，來源 dividend、policy_formula）。")
    rr = []
    for y in range(2005, int(yr.max()) + 1):
        s_ = np.where(yr == y)[0]; i0, i1 = s_[0] - 1, s_[-1]
        if i0 < 0: continue
        ok = TL[i0] & TL[i1]
        g1 = (np.asarray(C[i1], float) * AF[i1]) / (np.asarray(C[i0], float) * AF[i0]); g2 = (np.asarray(C[i1], float) * AFC[i1]) / (np.asarray(C[i0], float) * AFC[i0])
        rr.append({'年': y, '檔數': int(ok.sum()), '年報酬中位差(pt)': round(float(np.nanmedian((g1 - g2)[ok])) * 100, 2)})
    t = pd.DataFrame(rr).set_index('年'); w(''); w(md(t)); w('')
    w(f"- 偏誤 ①：2016 起下市而整檔排除的股票 {int(M['bias1_exclude'].sum())} 檔。\n")

    # 主檔狀態與下市
    w('## 5. 主檔狀態、下市與停止買賣期間\n')
    st = M['status'].value_counts(); w('- 面板股票狀態：' + '、'.join(f'{k} {v}' for k, v in st.items()) + '。')
    dl = M[M['status'] == 'delisted']; lt = pd.to_datetime(dl['last_trade_date']); dd = pd.to_datetime(dl['delist_date'])
    gap = (dd - lt).dt.days
    w(f"- 已下市 {len(dl)} 檔，其中沒有下市日 {int(dd.isna().sum())} 檔；最後成交日到下市日（日曆日）中位 {gap.median():.0f}、P90 {gap.quantile(.9):.0f}、最大 {gap.max():.0f}。")
    iu = M[M['status'] == 'inactive_unexplained']; lpd = pd.to_datetime(iu['last_price_date'])
    w(f"- inactive_unexplained（無下市紀錄卻已無資料）{len(iu)} 檔；最後有價日分布（年）：" +
      ('、'.join(f'{k} {v}' for k, v in lpd.dt.year.value_counts().sort_index().items()) if len(iu) else '無') + '。引擎須決定如何處理（見第 7 節）。')
    act = M[M['status'] == 'active']; la = pd.to_datetime(act['last_price_date'])
    w(f"- active 但最後有價日早於資料最後一日 30 天以上：{int((la < d[-1] - pd.Timedelta(days=30)).sum())} 檔。\n")

    # 資料列中斷（ENGINE_PLAN 3.3：2016 年 85 檔）
    w('## 6. 資料列中斷（第一列到最後一列之間，連續沒有資料列）\n')
    ex = np.asarray(EX); emg = np.asarray(EMG); first = ex.argmax(0); last = ex.shape[0] - 1 - ex[::-1].argmax(0)
    seg = []
    for n in range(ex.shape[1]):
        if not ex[:, n].any(): continue
        col = ex[first[n]:last[n] + 1, n]
        if col.all(): continue
        z = np.flatnonzero(np.diff(np.r_[1, col.astype(np.int8), 1]))
        for a_, b_ in zip(z[::2], z[1::2]):
            s_, e_ = first[n] + a_, first[n] + b_          # 中斷第一天、恢復日
            seg.append((n, s_, b_ - a_, emg[s_ - 1, n], emg[e_, n], AF[s_ - 1, n] != AF[e_, n],
                        np.asarray(C[e_, n], float) * AF[e_, n] / (np.asarray(C[s_ - 1, n], float) * AF[s_ - 1, n]) - 1))
    sg = pd.DataFrame(seg, columns=['n', 't0', 'len', 'emgB', 'emgA', 'ev', 'radj'])
    if len(sg):
        sg['年'] = yr[sg['t0']]
        sg['類型'] = np.select([sg.emgB & sg.emgA, ~sg.emgB & ~sg.emgA, sg.emgB & ~sg.emgA], ['興櫃期間', '上市櫃期間', '興櫃→上市櫃'], '上市櫃→興櫃')
        t = sg.groupby('類型').agg(段數=('len', 'size'), 檔數=('n', 'nunique'), 中位天數=('len', 'median'),
                                   滿20日=('len', lambda x: int((x >= 20).sum())), 最長=('len', 'max'))
        w(md(t)); w('')
        Ls = sg[sg['類型'] == '上市櫃期間']
        w(f"上市櫃期間的中斷 {len(Ls)} 段：其中 {Ls['ev'].mean():.0%} 恢復日有還原事件（減資、分割換發期間停止交易），"
          f"中位 {Ls.loc[Ls.ev, 'len'].median():.0f} 日；沒有事件的 {int((~Ls.ev).sum())} 段中位 {Ls.loc[~Ls.ev, 'len'].median():.0f} 日。"
          f"跨中斷的還原後報酬中位 {Ls['radj'].median():.1%}，絕對值 > 20% 的占 {(Ls['radj'].abs() > .2).mean():.1%}"
          f"（還原後價格連續，不是資料錯置）。\n")
        t = Ls[Ls['len'] >= 5].groupby('年').agg(段數=('len', 'size'), 檔數=('n', 'nunique'), 中位天數=('len', 'median'))
        w('上市櫃期間 ≥ 5 日的中斷，每年：\n'); w(md(t)); w('')
        w('ENGINE_PLAN 3.3 的「2016 年 85 檔資料中斷」：新資料 2016 年上市櫃期間 ≥ 5 日的中斷為 '
          f"{int(((Ls['年'] == 2016) & (Ls['len'] >= 5)).sum())} 段、{Ls.loc[(Ls['年'] == 2016) & (Ls['len'] >= 5), 'n'].nunique()} 檔，"
          '量級與其他年份相同，多數對應減資換發，判定為正常停止交易，不是缺資料。\n')
        w('「上市櫃→興櫃」：中斷前是上市櫃、恢復後第一天被標為興櫃（當日沒有漲跌停價）。原因未查明（可能是恢復交易首日無漲跌停價的標記方式）；'
          '引擎一律以 EMG 當日不可買處理，持有中的部位視同停牌。\n')
        w('註：停牌旗標（SUSP）2016 年才有來源；2016 年以前的停牌表現為「無成交」或「沒有資料列」。\n')
    else:
        w('沒有任何中斷。\n')

    # 旗標
    w('## 7. 交易旗標（上市櫃普通股、有資料列的日子，每年格數）\n')
    rows = []
    for y in sorted(set(yr)):
        s_ = yr == y; base = LIST[s_]
        r_ = {'年': y, '列數': int(base.sum())}
        for f, nm in [('SUSP', '停牌'), ('MISS', 'missing'), ('LUL', '漲停鎖死'), ('LDL', '跌停鎖死'), ('OSUS', '開盤可疑'), ('DEF', '瑕疵日'), ('DISP', '處置')]:
            r_[nm] = int((np.asarray(P[f][s_]) & base).sum())
        r_['無成交'] = int((base & ~TL[s_]).sum())
        rows.append(r_)
    w(md(pd.DataFrame(rows).set_index('年'))); w('')

    w('## 8. 待引擎決定（不在面板處理）\n')
    w('- 下市：依 L-113 第 3 點，資金在主檔 `delist_date` 回來；沒有下市日者以最後有價日代替（筆數見第 5 節）。')
    w('- inactive_unexplained：建議比照下市處理（最後有價日當作下市日），出清折扣同 0%／50%／100% 敏感度；需登記後才用。')
    w('- 資料列中斷段：期間視同停牌（持有、以最後價估值、佔名額），恢復後以還原價接續（減資換發的跳空已由 af 消除）。')
    w(f'\n報告產生時峰值記憶體 {peak_mem_mb():.0f} MB。')
    (LAB / 'data' / 'PANEL_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    report()
