# tw-data 研究端驗收檢查（只印彙總數字，不輸出任何原始資料）
# 用法：在 tw-data 資料夾（已 git checkout v1.1）旁邊執行：
#   python verify_twdata.py <tw-data 資料夾路徑>
# 需要 pandas、pyarrow。把印出的全部內容貼回對話。
import sys, traceback
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

R = Path(sys.argv[1] if len(sys.argv) > 1 else 'tw-data')
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
def sec(t): print('\n=== ' + t, flush=True)
def run(name, f):
    sec(name)
    try: f()
    except Exception: print('（此項失敗）'); traceback.print_exc(limit=1)

M = pd.read_parquet(R / 'derived/security_master.parquet')
YRS = range(2004, 2027)
PX = pd.concat([pd.read_parquet(R / f'derived/price_adj/{y}.parquet') for y in YRS], ignore_index=True)
TR = pd.concat([pd.read_parquet(R / f'derived/tradable/{y}.parquet') for y in YRS], ignore_index=True)
IX = pd.read_parquet(R / 'derived/index.parquet')
for d in (PX, TR, IX): d['date'] = pd.to_datetime(d['date'])
COMMON = set(M.loc[M['security_class'] == 'common', 'security_key'])

def s0():
    print('主檔欄位：', M.columns.tolist()); print('價格欄位：', PX.columns.tolist())
    print('旗標欄位：', TR.columns.tolist()); print('指數欄位：', IX.columns.tolist())
    print('價格筆數（2004 起）', len(PX), '｜旗標筆數', len(TR))
run('0. 結構', s0)

DF = PX.merge(TR.drop(columns=[c for c in ('stock_id',) if c in TR.columns]), on=['security_key', 'date'], how='left')
DF = DF[DF['security_key'].isin(COMMON)].sort_values(['security_key', 'date'])
DF['yr'] = DF['date'].dt.year
B = lambda c: DF[c].fillna(False).astype(bool) if c in DF.columns else pd.Series(False, index=DF.index)
DF['listed'] = ~B('emerging'); DF['trade'] = B('has_trade') & ~B('suspended') & (DF['close'] > 0)

def s1():
    g = IX.groupby('index_id')['date'].agg(['min', 'max', 'count']); print(g)
    val = [c for c in IX.columns if c not in ('index_id', 'date')]
    print('數值欄位：', val)
    for c in val:
        w = IX.pivot_table(index='date', columns='index_id', values=c)
        y = w.resample('YE').last().pct_change(); y.index = y.index.year
        print(f'\n[{c}] 每年報酬'); print((y.loc[2005:] * 100).round(1).to_string())
run('1. 指數涵蓋與年報酬（報酬指數減價格指數 ≈ 股利率，應約 2–4%）', s1)

def s2():
    k = M.loc[(M['stock_id'] == '2330') & (M['security_class'] == 'common'), 'security_key'].iloc[0]
    d = DF[DF['security_key'] == k].copy()
    d['rp'] = d['close'] / d['close'].shift(1) - 1; d['ra'] = (d['close'] * d['af']) / (d['close'] * d['af']).shift(1) - 1
    ev = d[(d['af'] != d['af'].shift(1)) & d['af'].shift(1).notna()]
    print('2330 還原因子變動日數', len(ev), '｜這些日子的未還原報酬中位數', f"{ev['rp'].median():.2%}", '｜還原後報酬中位數', f"{ev['ra'].median():.2%}")
    y = d.set_index('date')[['rp', 'ra']].add(1).groupby(lambda x: x.year).prod() - 1; y['差'] = y['ra'] - y['rp']
    print((y.loc[2005:] * 100).round(1).to_string()); print('2010 年日均成交量（股）', int(d.loc[d['yr'] == 2010, 'volume'].mean()))
    print('最新一日 af', d['af'].iloc[-1])
run('2. 抽查 2330 還原（差＝每年股利率，應約 2–4%；除息日還原後報酬應接近 0）', s2)

def s3():
    eom = DF[DF['listed'] & DF['trade']].groupby('yr')['security_key'].nunique()
    d = DF[DF['listed']].copy(); d['n'] = d.groupby('security_key')['trade'].cumsum()
    ye = d[d['date'] == d.groupby('yr')['date'].transform('max')]
    old = ye[ye['n'] >= 250].groupby('yr')['security_key'].nunique()
    print(pd.DataFrame({'當年有成交的上市櫃普通股': eom, '年底時已上市櫃且累積成交≥250日': old}).to_string())
run('3. 可交易股數（研究股池基礎）', s3)

def s4():
    d = DF[DF['listed']].copy()
    d['gap'] = ~d['trade']; grp = (d['gap'] != d.groupby('security_key')['gap'].shift()).cumsum()
    runs = d[d['gap']].groupby([d['security_key'], grp]).agg(start=('date', 'min'), end=('date', 'max'), n=('date', 'size'))
    long_ = runs[runs['n'] >= 20].reset_index()
    last = DF.groupby('security_key')['date'].max()
    long_['結局'] = np.where(long_['end'].values == last.reindex(long_['security_key']).values, '到資料最後（下市或至今）', '之後恢復交易')
    long_['yr'] = long_['start'].dt.year
    print('連續 ≥ 20 個交易日無法成交（停牌或無成交）的段數', len(long_), '｜涉及證券', long_['security_key'].nunique())
    print(pd.crosstab(long_['yr'], long_['結局']).to_string())
    print('段長度分位數（日）', long_['n'].quantile([.5, .9, .99]).round(0).to_dict())
run('4. 長期無法成交（對應 FinLab 版 X1 的 226 筆停牌部位）', s4)

def s5():
    dl = M[(M['security_class'] == 'common') & (M['status'] == 'delisted')].copy()
    dl['dy'] = pd.to_datetime(dl['delist_date']).dt.year
    print('下市普通股（依下市年）'); print(dl.groupby('dy').size().loc[2005:].to_string())
    lt = pd.to_datetime(dl['last_trade_date']); gap = (pd.to_datetime(dl['delist_date']) - lt).dt.days
    print('下市日 − 最後成交日（日曆日）分位數', gap.quantile([.5, .9, .99]).to_dict())
    d = DF[DF['security_key'].isin(dl['security_key']) & DF['trade']].copy()
    d['adj'] = d['close'] * d['af']; tail = d.groupby('security_key').tail(61)
    r60 = tail.groupby('security_key')['adj'].agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    print('下市股最後 60 個成交日報酬：中位數', f'{r60.median():.1%}', '｜P10', f'{r60.quantile(.1):.1%}', '｜樣本', len(r60))
    print('最後成交價 < 5 元的比例', f"{(d.groupby('security_key')['close'].last() < 5).mean():.0%}")
run('5. 下市股（決定回測中下市部位的處理方式）', s5)

def s6():
    d = DF[DF['listed'] & DF['trade']]
    t = d.groupby('yr').agg(成交日數=('trade', 'size'), 瑕疵日=('defect_day', lambda s: s.fillna(False).sum()),
                            開盤可疑=('open_suspect', lambda s: s.fillna(False).sum()))
    t['開盤可疑比例'] = (t['開盤可疑'] / t['成交日數']).map('{:.2%}'.format); print(t.loc[2005:].to_string())
run('6. 瑕疵日與開盤價可疑（上市櫃普通股、有成交日）', s6)

def s7():
    k = M.groupby('stock_id')['security_key'].nunique(); print('同代碼拆成多個證券的代碼數', int((k > 1).sum()))
    d = DF[['security_key', 'date', 'af']].dropna()
    r = d.groupby('security_key')['af'].pct_change().abs(); print('af 單日變動 > 50% 的筆數', int((r > 0.5).sum()))
    last = d.groupby('security_key')['af'].last(); print('最後一日 af ≠ 1 的證券數', int((last.round(6) != 1).sum()))
    print('price_status 分佈'); print(PX['price_status'].value_counts().to_string() if 'price_status' in PX.columns else '無此欄')
run('7. 代碼重用與還原因子', s7)
print('\n完成。', flush=True)
