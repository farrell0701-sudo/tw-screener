# -*- coding: utf-8 -*-
# FinLab 外部資料擷取：融資、法人期貨、海外市場、匯率
# 用法：在 FinLab 環境（已登入）執行本檔或貼到 notebook 一格執行。
# 產出：
#   ext_market.csv          每日一列的市場層級序列（主要檔案，請傳回）
#   ext_margin_stock.csv.gz 個股融資餘額（張）與融資使用率（選用，檔案較大）
#   ext_log.txt             每個資料集是否成功、欄位名稱、日期範圍（請一起傳回）
import gc, traceback
import numpy as np, pandas as pd
from finlab import data

LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

def tryget(keys):
    """依序嘗試多個資料集名稱，回傳第一個成功的 (key, DataFrame)。"""
    for k in keys:
        try:
            x = pd.DataFrame(data.get(k)); x.index = pd.to_datetime(x.index)
            log(f'[OK] {k}  shape={x.shape}  {x.index.min().date()}~{x.index.max().date()}  欄位前10={list(x.columns[:10])}')
            return k, x
        except Exception as e:
            log(f'[失敗] {k}: {type(e).__name__} {str(e)[:120]}')
    return None, None

# ---------- 0. 資料集探索（名稱不確定時用，結果寫進 log） ----------
for kw in ['融資', 'margin', '期貨', 'futures', '未平倉', 'world', '匯率', 'exchange', 'vix']:
    try:
        r = data.search(kw)
        log(f'[search] {kw}: {list(r)[:40] if r is not None else r}')
    except Exception as e:
        log(f'[search 不支援] {kw}: {type(e).__name__}')
        break

OUT = {}

# ---------- 1. 融資（個股加總成市場層級） ----------
_, bal = tryget(['margin_transactions:融資今日餘額'])
_, lim = tryget(['margin_transactions:融資限額'])
_, sbal = tryget(['margin_transactions:融券今日餘額'])
_, close = tryget(['price:收盤價'])
if bal is not None:
    bal = bal.apply(pd.to_numeric, errors='coerce')
    OUT['融資餘額_張'] = bal.sum(1, min_count=1)
    if close is not None:
        c = close.reindex(index=bal.index, columns=bal.columns).apply(pd.to_numeric, errors='coerce')
        OUT['融資餘額_億元'] = (bal * c * 1000).sum(1, min_count=1) / 1e8   # 1 張 = 1000 股，以收盤價估市值
        OUT['市場總市值代理_億元'] = None  # 佔位，下方無總市值資料時刪除
    if lim is not None:
        lim = lim.reindex(index=bal.index, columns=bal.columns).apply(pd.to_numeric, errors='coerce')
        OUT['融資使用率_整體'] = bal.sum(1, min_count=1) / lim.sum(1, min_count=1)
        use = bal / lim.replace(0, np.nan)
        OUT['融資使用率_個股中位'] = use.median(1)
        OUT['融資使用率>60%_家數比'] = (use > 0.6).sum(1) / use.notna().sum(1)
    if sbal is not None:
        sbal = sbal.reindex(index=bal.index, columns=bal.columns).apply(pd.to_numeric, errors='coerce')
        OUT['融券餘額_張'] = sbal.sum(1, min_count=1)
        OUT['券資比'] = OUT['融券餘額_張'] / OUT['融資餘額_張']
    try:
        keep = pd.concat({'bal': bal.stack(), 'use': (bal / lim.replace(0, np.nan)).stack() if lim is not None else bal.stack() * np.nan}, axis=1)
        keep.to_csv('ext_margin_stock.csv.gz', compression='gzip')
        log('[存檔] ext_margin_stock.csv.gz', keep.shape)
    except Exception as e:
        log('[個股融資存檔失敗]', e)
OUT.pop('市場總市值代理_億元', None)
del lim, sbal; gc.collect()

# 市場層級的融資餘額（若 FinLab 另有總表，作為交叉檢查）
for kk in [['margin_balance:融資券總餘額'], ['margin_balance:融資券']]:
    k, x = tryget(kk)
    if x is not None:
        log(f'[{k} 全部欄位]', list(x.columns))
        for cc in x.columns:
            OUT[f'融資總表_{cc}'] = pd.to_numeric(x[cc], errors='coerce')

# ---------- 2. 三大法人期貨與選擇權 ----------
FB = 'futures_institutional_investors_trading_summary'
for col in ['多空未平倉口數淨額', '多空未平倉契約金額淨額(千元)', '多方未平倉口數', '空方未平倉口數', '多空交易口數淨額']:
    k, x = tryget([f'{FB}:{col}'])
    if x is not None:
        log(f'[{k} 全部欄位]', list(x.columns)[:60])
        for cc in x.columns:
            OUT[f'期貨_{col}_{cc}'] = pd.to_numeric(x[cc], errors='coerce')
for kk in ['tw_option_put_call_ratio', 'tw_taifex_futures_institutional_breakdown', 'tw_taifex_futures_large_trader']:
    k, x = tryget([kk])
    if x is not None:
        log(f'[{k} 全部欄位]', list(x.columns)[:60])
        x.to_csv(f'ext_{kk}.csv', encoding='utf-8-sig')   # 原樣另存，欄位結構未知

# ---------- 3. 海外市場與 VIX ----------
k, w = tryget(['world_index:close', 'world_index:adj_close', 'world_index:收盤價'])
if w is not None:
    for sym in ['^GSPC', '^IXIC', '^SOX', '^VIX', '^DJI', '^N225', '^HSI', '^KS11', '^RUT', '^TWII']:
        if sym in w.columns:
            v = pd.to_numeric(w[sym], errors='coerce'); OUT[f'world_{sym}'] = v
            log(f'[world {sym}] 首筆非空 {v.first_valid_index()}  筆數 {v.notna().sum()}')
    log('[world_index 全部欄位]', list(w.columns))

# ---------- 4. 匯率（FinLab 無此資料集，保留嘗試） ----------
k, fx = tryget(['exchange_rate:收盤價', 'exchange_rate', 'fx:close', 'currency:close'])
if fx is not None:
    for cc in fx.columns:
        if any(t in str(cc).upper() for t in ['USD', 'JPY', '美元', '日圓', '日幣']):
            OUT[f'fx_{cc}'] = pd.to_numeric(fx[cc], errors='coerce')
    log('[匯率 全部欄位]', list(fx.columns))

# ---------- 5. 台股指數（對齊用） ----------
k, ix = tryget(['taiex_total_index:收盤指數', 'stock_index_price:收盤指數', 'benchmark_return:發行量加權股價報酬指數'])
if ix is not None:
    col = [c for c in ix.columns if '加權' in str(c) or 'TAIEX' in str(c).upper()]
    OUT['台股指數'] = pd.to_numeric(ix[col[0]] if col else ix.iloc[:, 0], errors='coerce')

# ---------- 存檔 ----------
DF = pd.DataFrame({k: v for k, v in OUT.items() if v is not None}).sort_index()
DF = DF.loc[DF.index >= '2004-01-01']
DF.to_csv('ext_market.csv', encoding='utf-8-sig')
log('[存檔] ext_market.csv', DF.shape, DF.index.min(), '~', DF.index.max())
log('[每欄非空筆數]', DF.notna().sum().to_dict())
open('ext_log.txt', 'w', encoding='utf-8').write('\n'.join(LOG))
print('完成：請傳回 ext_market.csv 與 ext_log.txt（ext_margin_stock.csv.gz 選用）')
