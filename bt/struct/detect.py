"""結構轉變辨識：C（打底區）→ B（首次突破）→ A（回測上緣、與 C 幾乎不重疊）
全部只用當天以前的資料（因果），可逐日掃描。"""
import numpy as np, pandas as pd

PAR = dict(
    base_min=40, base_max=160,      # C 區長度（交易日）
    base_q=(0.10, 0.90),            # C 區上下緣：最高價的 90 分位、最低價的 10 分位（忽略少數影線）
    base_width=0.10,                # C 區寬度 ÷ 中價 上限
    base_inside=0.80,               # C 區內收盤落在上下緣之間的比例下限（反覆重疊＝整理）
    a_len=5,                        # A 區長度
    a_width=0.07,                   # A 區寬度 ÷ 收盤 上限（小幅整理）
    a_ovC=0.25,                     # A 與 C 重疊 ÷ A 寬度 上限（幾乎不重疊）
    b_min=3, b_max=30,              # 突破日到今天的天數
    a_retest=0.04,                  # A 低點距 C 上緣（上方）不超過 4%：真的有回測
)


def find_base(H, L, C, end, p=PAR):
    """在 end（不含）之前找最長的整理區 C；回傳 (start, top, bot) 或 None"""
    best = None
    for n in range(p['base_min'], min(p['base_max'], end) + 1, 5):
        h = H[end - n:end]; l = L[end - n:end]; c = C[end - n:end]
        top = np.quantile(h, p['base_q'][1]); bot = np.quantile(l, p['base_q'][0])
        mid = (top + bot) / 2
        if (top - bot) / mid > p['base_width']:
            break                                   # 再往前延伸只會更寬
        if ((c <= top) & (c >= bot)).mean() >= p['base_inside']:
            best = (end - n, top, bot)
    return best


def scan(df, p=PAR):
    """df: 還原 High/Low/Close（index 為日期）。回傳每個「結構完成」事件"""
    H, L, C = df.High.values, df.Low.values, df.Close.values
    ev = []; last_bo = -999
    for t in range(p['base_min'] + p['b_max'] + p['a_len'], len(C)):
        a0 = t - p['a_len'] + 1
        hA, lA = H[a0:t + 1].max(), L[a0:t + 1].min()
        if (hA - lA) / C[t] > p['a_width']:
            continue
        # 往回找突破日 s：s 之前是整理區 C，s 當天收盤首次站上 C 上緣
        for s in range(a0 - 1, max(a0 - p['b_max'], p['base_min']), -1):
            base = find_base(H, L, C, s, p)
            if base is None:
                continue
            b0, top, bot = base
            if not (C[s] > top and C[s - 1] <= top):
                continue
            ovC = max(0.0, min(hA, top) - max(lA, bot)) / (hA - lA + 1e-9)
            if a0 - s < p['b_min']:
                break
            hB = H[s:a0].max()
            ovB = max(0.0, min(hA, hB) - lA) > 0                 # A 與 B 有交集
            pullback = hA <= hB                                   # A 沒有再創高：是整理不是續攻
            retest = lA <= top * (1 + p['a_retest'])              # A 低點回到 C 上緣附近
            if ovC <= p['a_ovC'] and ovB and pullback and retest and lA >= bot and s != last_bo:
                ev.append(dict(date=df.index[t], c_start=df.index[b0], c_end=df.index[s - 1], b_day=df.index[s],
                               c_top=top, c_bot=bot, a_hi=hA, a_lo=lA, ovC=ovC,
                               c_len=s - b0, b_len=a0 - s, b_hi=hB, c_width=(top - bot) / ((top + bot) / 2)))
                last_bo = s
            break
    return pd.DataFrame(ev)
