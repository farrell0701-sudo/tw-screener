# 步驟 3 檢查：core/signals.py
# A. 從 lab/finlab/fl_small_report.py 原文抽出第 92–156 行（T_ld、可交易池、X1、V4.1），在同一份輸入（全市場）上原封不動執行，
#    逐格比對 RKX、RKV 與各中間遮罩。判準：完全一致（排名值相同、空值位置相同）。
# B. 量級：每日池子大小、剔除檔數、事件數，與舊紀錄比較（L-106 基準事件數 05-15 2,397、16-26 2,693；L-101 過熱 20%、最陡 17–18%）。
# 用法：python lab/core/test_signals.py [compat|fix]
import sys
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
from signals import compute, MODES

try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
mode = sys.argv[1] if len(sys.argv) > 1 else 'compat'
md = MODES[mode]
P = load_panel(); F = load_feat('AF_C' if md['bias2'] else 'AF'); M = P['meta']; dates = P['dates']
r = compute(mode, P=P, F=F); cols = r['_dbg']['cols']
g = lambda a: np.asarray(a)[:, cols]
names = M['security_key'].to_numpy()[cols]
DFa = lambda a: pd.DataFrame(a, index=dates, columns=names)

# ---------- A. 參考實作原文
src = (LAB / 'finlab' / 'fl_small_report.py').read_text(encoding='utf-8').splitlines()
i = next(k for k, s in enumerate(src) if s.startswith('DF = lambda'))
j = next(k for k, s in enumerate(src) if s.startswith('RKV = score.where'))
code = '\n'.join(s for s in src[i:j + 1] if not s.lstrip().startswith('print('))
a_ = next(k for k, s in enumerate(src) if s.startswith('def iakf_down'))
b_ = next(k for k, s in enumerate(src) if s.startswith('    return pd.Series(dn, index=x.index)'))
f32 = np.float32
ind_vals = (M['ind_pk'] if md['ind'] == 'pk' else M['industry']).to_numpy()[cols]
ns = dict(np=np, pd=pd, f32=f32, swv=swv, gc=__import__('gc'), dates=dates, T_=len(dates), N_=len(cols),
          Cn=g(F['CA']).astype(f32), VV=g(F['VALID']), ATR=g(F['ATR']), NUM=g(F['NUM']), AMT=g(F[md['amt']]),
          R20=g(F['R20']), S60=g(F['S60']), RRT=g(F['RRT']), EV1=g(F['EV1']), EV2=g(F['EV2']), OLD=g(F['OLD']),
          H=DFa(g(F['HA'])), L=DFa(g(F['LA'])),   # 參考程式在此區段內重算 S60、RRT（等於再驗一次）
          C=DFa(g(F['CA']).astype(f32)), O=DFa(g(F['OA'])), RC=DFa(g(P['C'])), ROP=DFa(g(P['O'])),
          RH=DFa(np.fmax(np.fmax(g(P['H']), g(P['O'])), g(P['C']))), RL=DFa(np.fmin(np.fmin(g(P['L']), g(P['O'])), g(P['C']))),
          ix=P['index']['TAIEX_P'].reindex(dates).ffill(),
          ind=pd.Series([v if isinstance(v, str) else None for v in ind_vals], index=names))
exec('\n'.join(src[a_:b_ + 1]), ns)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exec(code, ns)

D = r['_dbg']; res = {}
def same(nm, a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype == bool or b.dtype == bool: res[nm] = int((a.astype(bool) != b.astype(bool)).sum())
    else:
        na, nb = np.isnan(a.astype(float)), np.isnan(b.astype(float))
        res[nm] = int((na != nb).sum() + (np.abs(np.where(na | nb, 0, a - b)) > 1e-9).sum())
same('T_ld', np.nan_to_num(D['Tr'], nan=-9e9), np.nan_to_num(ns['Tr'], nan=-9e9))
same('可交易池 U', D['U'], ns['U'])
same('強勢池 pool', D['pool'], ns['pool'])
same('過熱 hot', D['hot'], ns['hot'])
same('最陡 steep', D['steep'], ns['steep'])
same('X1 排名 RKX', g(r['RKX']), ns['RKX'])
same('大盤 < 200MA', r['daily']['below'].values, ns['below'])
same('空頭模式', r['daily']['bear_mode'].values, ns['MD'])
same('IAKF 關閉', r['daily']['off'].values, ns['OFF'])
same('V4.1 強勢池', D['strong'], ns['strong'].values)
same('V4.1 事件 sig0', D['sig0'], ns['sig0'])
same('V4.1 訊號', D['signal'], ns['signal'])
same('V4.1 排名 RKV', g(r['RKV']), ns['RKV'])
print(f'模式 {mode}｜{len(cols)} 檔｜{len(dates)} 日')
bad = [k for k, v in res.items() if v]
for k, v in res.items(): print(f"{'通過' if v == 0 else '不通過'}  A 參考 {k}：不同格數 {v}")

# ---------- B. 量級
d = r['daily']; seg = {'2005–15': ('2005-01-01', '2015-12-31'), '2016–26': ('2016-01-01', '2026-12-31')}
rows = []
for nm, (a, b) in seg.items():
    x = d.loc[a:b]
    rows.append({'期間': nm, '可交易池/日': round(x.n_U.mean()), '強勢池/日': round(x.n_pool.mean(), 1),
                 '過熱占池': f'{x.n_hot.sum() / x.n_pool.sum():.1%}', '最陡占池': f'{x.n_steep.sum() / x.n_pool.sum():.1%}',
                 'X1 候選/日': round(x.n_candX.mean(), 1), 'V4.1 事件（剔深回撤前）': int(x.n_sig0.sum()),
                 '剔深回撤': f'{x.n_deep.sum() / max(x.n_sig0.sum(), 1):.1%}', 'V4.1 訊號': int(x.n_signal.sum()),
                 '空頭模式': f'{x.bear_mode.mean():.1%}', 'IAKF 關閉': f'{x.off.mean():.1%}', '< 200MA': f'{x.below.mean():.1%}'})
print(pd.DataFrame(rows).set_index('期間').T.to_string())
print('對照：L-106 基準 V4.1 事件數 2,397（05-15）、2,693（16-26）；L-101 過熱 20%、最陡 17–18%；L-096 剔深回撤 6.8%／8.1%。')
print('結論：' + ('A 全部通過' if not bad else f'A 有 {len(bad)} 項不通過：{bad}'))
