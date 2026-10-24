"""全市場寬表（日期 × 代號），含 2016 後下市股。
價格：C/O/H/L 為還原價（算報酬與技術指標用），RC 為 FinMind 真正未還原收盤，V 為成交股數。
輸出 /home/claude/bt/fac/panel.pkl"""
import os, sys
os.environ.setdefault('RAWSRC', 'finmind'); os.environ.setdefault('DELIST', '1')
sys.path.insert(0, '/home/claude/bt')
import numpy as np, pandas as pd
from load import load_all

OUT = '/home/claude/bt/fac/panel.pkl'


def build():
    lst, tx, cal, S, _ = load_all()
    codes = [c for c in S if len(c) == 4 and c.isdigit() and not c.startswith('0')]
    P = {}
    for k in ('O', 'H', 'L', 'C', 'V', 'RC'):
        P[k] = pd.DataFrame({c: S[c][k] for c in codes}).reindex(cal).astype('float64')
    P['AMT'] = P['RC'] * P['V']                       # 成交金額（元）
    meta = lst.set_index('code').reindex(codes)
    P['industry'] = meta.industry
    P['market'] = meta.market
    P['taiex'] = tx.Close.reindex(cal)
    return P


def load_panel():
    if os.path.exists(OUT):
        return pd.read_pickle(OUT)
    P = build(); pd.to_pickle(P, OUT); return P


if __name__ == '__main__':
    P = load_panel()
    C = P['C']
    print('日期', C.index[0].date(), '~', C.index[-1].date(), C.shape)
    print('每日有價股票數（中位）', int(C.notna().sum(1).median()), '下市股', int((P['market'] == 'DL').sum()))
