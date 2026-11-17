# 小資金合併版（1 倍）完整回測報告 —— FinLab 版（2026-11-17）
# 規則：2026-11-15 定版。X1 6 檔等權 60% ＋ V4.1 4 檔等權 40%，每月再平衡，不開槓桿。
#   X1：強勢池、剔過熱（20 日漲幅池內前 20%）、剔最陡（強勢池中 EMA200 近 60 日上升 ≥ 5% 者的前 20%）。
#   V4.1：剔深回撤（60 日波段回撤比 ≥ 23.6%，波段漲幅 < 15% 視為無波段），已刪除「近 5 日一字跌停排除」。
# 注意：剔最陡、剔深回撤的原始程式已遺失，此處依規則文件重寫，可能與舊引擎有細微差異。
# 用法：在已登入 FinLab 的 notebook 貼上整格執行（約 20–40 分鐘），把印出的全部內容貼回。
#       最後會存兩個檔：small_merge_daily.csv、small_merge_trades.csv（請勿上傳到公開 repo）。
import gc
for _n in [n for n in list(globals()) if not n.startswith('_') and n not in ('gc', 'In', 'Out', 'get_ipython', 'exit', 'quit')]:
    del globals()[_n]
gc.collect()
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from finlab import data
FEE, TAX, SLIP, RATE, NPATH = 0.00085, 0.003, 0.0015, 0.0645, 20
MI = RATE / 252; f32 = np.float32
sc = data.get('security_categories'); sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
bad = set(sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str)); del sc
def load(key, cols=None):
    x = data.get(key)
    if cols is None:
        cols = [c for c in x.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('0') and str(c) not in bad]
    y = x.reindex(columns=cols).astype(f32); del x; gc.collect(); return y, cols
RC, cols = load('price:收盤價')
C, _ = load('etl:adj_close', cols); RH, _ = load('price:最高價', cols); RL, _ = load('price:最低價', cols)
ROP, _ = load('price:開盤價', cols); V, _ = load('price:成交股數', cols)
RC, RH, RL, ROP, V = [x.reindex(C.index) for x in (RC, RH, RL, ROP, V)]
_cnt = pd.Series((pd.DataFrame(C).notna().values & (pd.DataFrame(V).fillna(0).values > 0)).sum(1), index=C.index)
_last = _cnt[_cnt >= 0.5 * _cnt[_cnt > 0].median()].index[-1]
C, RC, RH, RL, ROP, V = [x.loc[:_last] for x in (C, RC, RH, RL, ROP, V)]
RH = np.fmax(np.fmax(RH, ROP), RC); RL = np.fmin(np.fmin(RL, ROP), RC)
k_ = C / RC; O, H, L = ROP * k_, RH * k_, RL * k_; del k_
r_ = C / C.ffill().shift(1) - 1
jump = (r_.abs() > 0.11) & (C.notna().cumsum() > 5) & C.notna()
fix = (1 + r_.where(jump, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0).astype(f32)
O, H, L, C = O * fix, H * fix, L * fix, C * fix; del r_, jump, fix; gc.collect()
VALID = C.notna() & (V.fillna(0) > 0)
dates = C.index; T_, N_ = C.shape
print('資料', dates[0].date(), '～', dates[-1].date(), '｜股票', N_, flush=True)
def idx_series(key, col):
    s = pd.DataFrame(data.get(key)); s = s[col] if col in s.columns else s.iloc[:, 0]; s.index = pd.to_datetime(s.index); return s.astype(float).dropna()
best = None
for key, col in (('stock_index_price:收盤指數', '上市發行量加權股價指數'), ('benchmark_return:發行量加權股價報酬指數', None), ('taiex_total_index:收盤指數', None)):
    try:
        s_ = idx_series(key, col)
        if best is None or s_.index[0] < best[1].index[0]: best = (key, s_)
    except Exception as e:
        print('大盤資料讀取失敗', key, str(e)[:60])
ix = best[1].reindex(dates).ffill(); ix_src = best[0]
if best[1].index[0] > dates[0] + pd.Timedelta(days=30):
    w_ = pd.DataFrame((RC * V).values, index=dates).shift(1).rolling(20, min_periods=5).mean()
    rr_ = pd.DataFrame((RC / RC.shift(1) - 1).values, index=dates).clip(-0.11, 0.11)
    pr_ = (rr_ * w_).sum(axis=1) / w_.where(rr_.notna()).sum(axis=1); s0_ = best[1].index[0]
    px_ = (1 + pr_.fillna(0)).cumprod(); px_ = px_ / px_.loc[px_.index[px_.index.get_indexer([s0_], method='bfill')[0]]] * best[1].iloc[0]
    ix = ix.where(ix.index >= s0_, px_); del w_, rr_, pr_, px_
print('大盤序列', ix_src, flush=True)
PK = "光電業:2323,2349,2374,2393,2406,2409,2426,2429,2438,2466,2486,2489,2491,3008,3019,3024,3031,3038,3049,3050,3051,3059,3066,3128,3149,3168,3230,3297,3339,3356,3362,3406,3434,3437,3441,3455,3481,3490,3504,3516,3523,3531,3535,3543,3563,3576,3591,3615,3622,3623,3630,3666,3673,3691,3714,3718,4729,4933,4934,4935,4942,4949,4956,4960,4972,4976,4995,5220,5230,5234,5243,5244,5245,5251,5315,5392,5484,6116,6120,6164,6167,6168,6176,6209,6222,6225,6226,6234,6244,6246,6278,6405,6419,6443,6456,6477,6498,6517,6556,6560,6668,6706,6742,6859,6899,6916,7402,7753,8049,8064,8069,8104,8105,8111,8215,8240|其他業:1342,1416,1435,1437,1443,1516,1584,2221,2348,2496,2514,2614,2724,2904,3040,3284,4154,4430,4529,4541,4556,5209,5276,5284,5314,5345,5398,5450,5481,5530,5604,5871,6179,6184,6199,6464,6504,6585,6592,6625,6655,6721,6881,6901,6904,6914,6952,6957,6958,7777,7855,8033,8342,8354,8401,8404,8411,8421,8426,8435,8437,8442,8444,8463,8466,8481,8488,8489,8905,8906,8916,8921,8929,8935,8936,8937,8942,9902,9905,9907,9917,9919,9925,9927,9928,9929,9933,9938,9939,9940,9941,9942,9944,9945|其他電子業:1785,2312,2317,2354,2359,2360,2373,2390,2404,2423,2433,2459,2461,2464,2474,2477,2482,2488,3018,3030,3043,3093,3219,3285,3289,3303,3305,3324,3373,3402,3465,3498,3508,3518,3541,3552,3580,3587,3617,3628,3663,3665,4554,4577,4585,4588,4760,5225,5452,5489,5493,5536,6139,6146,6151,6192,6196,6201,6215,6283,6409,6438,6512,6558,6613,6654,6658,6667,6691,6698,6722,6725,6735,6739,6743,6830,6840,6855,6877,6903,6983,7631,7703,7728,7792,7825,8021,8047,8085,8092,8201,8431,8455,8499|化學工業:1708,1709,1710,1711,1712,1713,1714,1717,1718,1721,1722,1723,1725,1726,1727,1730,1732,1735,1742,1773,1776,3430,4706,4707,4711,4714,4716,4720,4721,4722,4739,4741,4754,4755,4763,4764,4766,4767,4768,4770,4772,6509|半導體業:1595,2302,2303,2329,2330,2337,2338,2340,2342,2344,2351,2363,2369,2379,2388,2401,2408,2434,2436,2441,2449,2451,2454,2458,2481,3006,3014,3016,3034,3035,3041,3094,3105,3122,3131,3135,3141,3150,3169,3178,3189,3227,3228,3257,3259,3260,3264,3265,3268,3317,3372,3374,3413,3438,3443,3450,3467,3527,3529,3530,3532,3545,3555,3556,3567,3581,3583,3588,3592,3661,3675,3680,3686,3707,3711,4749,4919,4923,4951,4952,4961,4966,4967,4968,4971,4973,4991,5222,5236,5269,5272,5274,5285,5299,5302,5344,5347,5351,5425,5443,5468,5471,5483,5487,6103,6104,6129,6138,6147,6182,6187,6202,6208,6223,6229,6233,6237,6239,6243,6257,6261,6271,6291,6411,6415,6423,6435,6451,6462,6485,6488,6494,6510,6515,6525,6526,6531,6532,6533,6548,6552,6568,6573,6640,6643,6651,6679,6683,6684,6693,6695,6708,6716,6719,6720,6732,6756,6770,6788,6789,6799,6823,6829,6854,6895,6907,6909,6921,6937,6953,6962,6996,7556,7704,7712,7730,7734,7749,7751,7768,7769,7770,7772,7810,7822,7828,7856,8016,8024,8028,8040,8054,8081,8086,8088,8091,8102,8110,8131,8150,8162,8227,8261,8271,8277,8299,8383|塑膠工業:1301,1303,1304,1305,1307,1308,1309,1310,1312,1313,1314,1315,1321,1323,1324,1325,1326,1337,1340,1341,4303,4304,4305,4306,9950|居家生活:2062,2230,2916,2924,2937,2938,2941,2947,2948,3067,3171,3557,4609,4702,5902,5903,5904,6195,6616,6629,6671,6728,6754,6807,6968,7782,7794,8066,8433,8464,8482,8941,9911,9924,9934,9935|建材營造業:1316,1436,1438,1439,1442,1453,1456,1472,1805,1808,2442,2501,2504,2505,2506,2509,2511,2515,2516,2520,2524,2527,2528,2530,2534,2535,2536,2537,2538,2539,2540,2542,2543,2545,2546,2547,2548,2596,2597,2718,2923,3052,3056,3188,3266,3313,3489,3512,3521,3703,4113,4416,4907,5206,5213,5324,5455,5508,5511,5512,5514,5515,5516,5519,5520,5521,5522,5523,5525,5529,5531,5533,5534,5543,5546,5547,5548,6171,6177,6186,6198,6212,6219,6264,8080,8424,9906,9946|數位雲端:2640,2949,3085,3130,3687,5278,5287,5321,6165,6236,6614,6689,6690,6741,6763,6811,6865,6870,6902,6906,6910,6925,6997,7547,7714,7721,7722,7738,7747,7765,7805,7823,7835,7839,8044,8454,8472,8477,8487,8932|文化創意業:2926,3064,3083,3086,3293,3546,3629,4806,4946,5263,5478,6101,6111,6144,6169,6180,6294,6482,6542,6596,6856,7584,8446,8450,8923,9949|橡膠工業:2101,2102,2103,2104,2105,2106,2107,2108,2109,2114,6582|水泥工業:1101,1102,1103,1104,1108,1109,1110|汽車工業:1319,1338,1339,1512,1521,1522,1524,1525,1533,1536,1563,1568,1587,2115,2201,2204,2206,2207,2227,2228,2231,2233,2236,2237,2239,2241,2243,2247,2248,2250,2254,2258,2497,3346,3717,4551,4557,4569,4581,6605,6988,7732,7736,7821|油電燃氣業:2616,6505,8908,8917,8926,8927,8931,9908,9918,9926,9931,9937|玻璃陶瓷:1802,1806,1809,1810,1817|生技醫療業:1565,1707,1720,1731,1733,1734,1752,1760,1762,1777,1781,1783,1784,1786,1788,1789,1795,1799,1813,3118,3164,3176,3205,3218,3705,3716,4102,4104,4105,4106,4107,4108,4109,4111,4114,4116,4119,4120,4121,4123,4126,4127,4128,4129,4131,4133,4137,4138,4139,4142,4147,4148,4153,4155,4157,4160,4161,4162,4163,4164,4166,4167,4168,4169,4173,4174,4175,4178,4183,4188,4190,4192,4195,4198,4726,4728,4735,4736,4737,4743,4744,4745,4746,4747,4771,4905,4911,5312,6130,6242,6431,6446,6461,6469,6472,6491,6492,6496,6499,6523,6527,6534,6535,6541,6547,6550,6569,6574,6576,6589,6598,6612,6615,6617,6620,6637,6645,6649,6657,6661,6662,6666,6703,6712,6730,6733,6762,6767,6782,6785,6794,6796,6838,6841,6844,6861,6872,6875,6885,6918,6919,6929,6931,6934,6936,6945,6949,6955,6986,7713,7799,7803,7814,7827,8279,8403,8409,8432,8436|紡織纖維:1402,1409,1410,1413,1414,1417,1418,1419,1423,1434,1440,1441,1444,1445,1446,1447,1449,1451,1452,1454,1455,1457,1459,1460,1463,1464,1465,1466,1467,1468,1470,1473,1474,1475,1476,1477,4401,4402,4406,4413,4414,4417,4420,4426,4432,4433,4438,4439,4440,4441,4442,6506|綠能環保:2072,3073,3551,3708,3713,4582,4924,5205,5292,5432,6581,6624,6641,6692,6771,6803,6869,6873,6887,6894,6923,6944,6947,6951,6969,6971,6994,7610,7715,7740,7786,7818,7820,7842,8087,8171,8341,8390,8422,8423,8438,8440,8473,8476,9930,9955|航運業:2208,2601,2603,2605,2606,2607,2608,2609,2610,2611,2612,2613,2615,2617,2618,2630,2633,2634,2636,2637,2641,2642,2643,2645,2646,5601,5603,5607,5608,5609,6753,6757,7716,8367|觀光餐旅:1259,1268,2701,2702,2704,2705,2706,2707,2712,2719,2722,2723,2726,2727,2729,2731,2732,2734,2736,2739,2740,2743,2745,2748,2751,2752,2753,2754,2755,2756,3252,3522,4419,4530,4804,5301,5364,5701,5703,5704,5706,5905,6961,7705,7708,7723,7757,7760,8077,8940,9943|貿易百貨業:2901,2903,2905,2906,2908,2910,2911,2912,2913,2915,2929,2939,2945,4807,5906,5907,8429,8443|資訊服務業:2427,2453,2468,2471,2480,3029,3147,3158,3570,4953,4994,5201,5202,5203,5210,5211,5212,5310,5403,5410,6112,6123,6140,6148,6163,6183,6214,6221,6231,6240,6516,6590,6593,6697,6751,6752,6791,6874,6884,7767,7819,8099,8272,8284,8416|農業科技業:1240,4171,6508,6578|通信網路業:2314,2321,2332,2345,2412,2419,2424,2439,2444,2450,2455,2485,2498,3025,3027,3045,3047,3062,3081,3095,3138,3152,3163,3221,3234,3306,3311,3363,3380,3419,3447,3466,3491,3499,3558,3564,3596,3632,3664,3669,3672,3684,3694,3704,4903,4904,4906,4908,4909,4977,4979,5353,5388,6109,6136,6142,6143,6152,6170,6190,6216,6218,6241,6245,6263,6285,6416,6417,6426,6442,6465,6470,6486,6530,6546,6561,6588,6674,6792,6863,7689,7717,7812,8011,8034,8045,8048,8059,8089,8097,8101,8176|造紙工業:1903,1904,1905,1906,1907,1909,6790|運動休閒:1432,1593,1598,1736,2762,4536,5306,5348,6670,6768,6804,6890,6965,7811,8462,8467,8478,8924,8928,8933,8938,9802,9904,9910,9914,9921,9960|金融保險業:2801,2812,2816,2820,2832,2834,2836,2838,2845,2849,2850,2851,2852,2855,2880,2881,2882,2883,2884,2885,2886,2887,2889,2890,2891,2892,2897,5864,5876,5878,5880,6005,6015,6016,6020,6021,6023,6024,6026,6028|鋼鐵工業:2002,2006,2007,2008,2009,2010,2012,2013,2014,2015,2017,2020,2022,2023,2024,2025,2027,2028,2029,2030,2031,2032,2033,2034,2035,2038,2063,2064,2065,2069,2073,2211,3004,4950,5007,5009,5011,5013,5014,5015,5016,5538,6248,8349,8415,8930,9958,9962|電器電纜:1603,1604,1605,1608,1609,1611,1612,1614,1615,1616,1617,1618,1623,1626,2061,4930,5283|電子通路業:2347,2414,2430,3010,3028,3033,3036,3048,3054,3055,3209,3224,3232,3312,3360,3444,3528,3702,5434,6113,6118,6154,6189,6227,6265,6270,6281,6474,6776,6908,8032,8067,8068,8070,8072,8084,8096,8112|電子零組件業:1336,1471,1582,1815,2059,2308,2313,2316,2327,2328,2355,2367,2368,2375,2383,2385,2392,2402,2413,2415,2420,2421,2428,2431,2440,2457,2460,2462,2467,2472,2476,2478,2483,2484,2492,2493,3003,3011,3015,3021,3023,3026,3032,3037,3042,3044,3058,3078,3090,3092,3114,3115,3191,3206,3207,3217,3229,3236,3276,3288,3290,3294,3296,3308,3310,3321,3322,3332,3338,3354,3357,3376,3388,3390,3432,3484,3485,3492,3501,3511,3520,3526,3533,3537,3548,3550,3593,3597,3605,3607,3609,3624,3631,3645,3646,3653,3679,3689,3710,3715,4542,4545,4912,4915,4927,4939,4943,4958,4974,4989,4999,5227,5228,5291,5309,5328,5340,5355,5439,5457,5460,5464,5469,5475,5488,5498,6108,6114,6115,6124,6126,6127,6133,6134,6141,6153,6155,6156,6158,6173,6174,6175,6185,6191,6194,6197,6203,6204,6205,6207,6210,6213,6217,6220,6224,6259,6266,6269,6272,6274,6275,6279,6282,6284,6290,6292,6412,6418,6432,6449,6538,6584,6597,6642,6664,6672,6715,6727,6761,6781,6805,6821,6834,6835,6862,6913,6924,6967,7744,7788,7795,8038,8039,8042,8043,8046,8071,8074,8093,8103,8109,8121,8147,8155,8182,8213,8249,8289,8291,8358|電機機械:1503,1504,1506,1513,1514,1515,1517,1519,1526,1527,1528,1529,1530,1531,1532,1535,1537,1538,1539,1540,1541,1558,1560,1570,1580,1583,1586,1589,1590,1591,1597,1599,2049,2066,2067,2070,2235,2371,3162,3167,3226,3379,3685,4502,4503,4506,4510,4513,4523,4526,4527,4528,4532,4533,4534,4535,4538,4540,4543,4549,4550,4552,4555,4558,4560,4561,4562,4563,4564,4566,4568,4571,4572,4576,4580,4583,4584,4590,5288,5381,6122,6125,6425,6603,6604,6606,6609,6843,6982,7642,7709,7718,7750,8027,8083,8107,8222,8255,8374,8996,9951|電腦及週邊設備業:1569,2301,2305,2324,2331,2352,2353,2356,2357,2362,2364,2365,2376,2377,2380,2382,2387,2395,2397,2399,2405,2417,2425,2432,2465,2495,3002,3005,3013,3017,3022,3046,3057,3060,3071,3088,3211,3213,3231,3272,3287,3323,3325,3349,3416,3479,3483,3494,3515,3540,3577,3594,3611,3625,3652,3693,3701,3706,3709,3712,4916,4931,4938,5215,5223,5258,5289,5356,5386,5426,5438,5465,5474,5490,6117,6121,6128,6150,6160,6161,6166,6188,6206,6228,6230,6235,6276,6277,6414,6441,6570,6577,6579,6591,6669,6680,6831,6922,6928,6933,7711,8050,8076,8114,8163,8210,8234,8410,9912|食品工業:1201,1203,1210,1213,1215,1216,1217,1218,1219,1220,1225,1227,1229,1231,1232,1233,1234,1235,1236,1256,1264,1294,1295,1702,1737,1780,1796,4205,4207,6846,7743,7780,7791,7806"
LOC = {c: k for blk in PK.split('|') for k, cs in [blk.split(':')] for c in cs.split(',')}
ind = pd.Series({c: LOC.get(c) for c in cols})
def iakf_down(x, smooth=250, nlook=10, thr=2.2, rej=0.05, decay=0.9):
    p_ = x.values.astype(float); ratio = (2.0 / (smooth + 1)) ** 2; a = 2.0 / (nlook + 1); thr2 = thr ** 2
    dn = np.zeros(len(p_), bool); started = False
    for t, p in enumerate(p_):
        if np.isnan(p): continue
        if not started:
            est, S, mS, R, calm, ev = p, 1e-4, 0.0, 1e-4, 1e-4, 1e-4; Q = 1e-4 * ratio; started = True
        pv = ev + Q; sur = p - est; cs = sur - mS; an = cs ** 2 / S > thr2
        mS = (1 - a) * mS + a * sur; S = (1 - a) * S + a * cs ** 2
        R = rej * cs ** 2 + R if an else (1 - decay) * S + decay * R
        calm = (1 - decay) * S + decay * calm
        K = pv / (pv + R); est = est + K * sur; ev = (1 - K) * pv; Q = S * ratio
        dn[t] = an and cs < 0
    return pd.Series(dn, index=x.index)
# ---- 個股指標
ATR = np.full((T_, N_), np.nan, f32); NUM = ATR.copy(); AMT = ATR.copy(); R20 = ATR.copy()
EV1 = np.zeros((T_, N_), bool); EV2 = np.zeros((T_, N_), bool); OLD = np.zeros((T_, N_), bool)
Cn, Hn, Ln, VV = C.values, H.values, L.values, VALID.values; An = (RC * V).values
for j in range(N_):
    ii = np.where(VV[:, j])[0]
    if len(ii) < 30: continue
    s = pd.DataFrame({'C': Cn[ii, j].astype(float), 'H': Hn[ii, j].astype(float), 'L': Ln[ii, j].astype(float), 'A': An[ii, j].astype(float)})
    pc = s.C.shift(1); tr = np.maximum(s.H - s.L, np.maximum((s.H - pc).abs(), (s.L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    m20 = s.C.rolling(20).mean(); sd20 = s.C.rolling(20).std(); up = m20 + 2 * sd20; bw = 4 * sd20 / m20
    rng_ = (s.H.rolling(20).max() - s.L.rolling(20).min()) / atr
    ATR[ii, j] = atr.values; NUM[ii, j] = (s.C.ewm(span=12, adjust=False).mean() - s.C.ewm(span=200, adjust=False).mean()).values
    AMT[ii, j] = s.A.rolling(20, min_periods=15).mean().values; R20[ii, j] = (s.C / s.C.shift(20) - 1).values
    EV1[ii, j] = ((bw.rolling(10).min().shift(1) <= bw.rolling(120, min_periods=100).quantile(0.2).shift(1)) & (s.C > up) & (pc <= up.shift(1))).values
    EV2[ii, j] = ((rng_.shift(1) <= rng_.rolling(250, min_periods=150).quantile(0.3).shift(1)) & (s.C > s.H.rolling(20).max().shift(1))).values
    OLD[ii[249:], j] = True
del An; gc.collect(); print('個股指標完成', flush=True)
DF = lambda a: pd.DataFrame(a, index=dates, columns=C.columns)
floor = np.nanquantile(np.where(VV, ATR / Cn, np.nan), 0.05, axis=1).astype(f32)
DEN = np.maximum(ATR, Cn * floor[:, None]); Tr = NUM / DEN
DP = np.full((T_, N_), np.nan, f32)
for j in range(N_):
    ii = np.where(VV[:, j])[0]
    if len(ii) > 20: DP[ii[20:], j] = DEN[ii[:-20], j]
atr_ok = DEN < DP; del DP
amt_q = np.nanquantile(np.where(VV & OLD, AMT, np.nan), 0.20, axis=1)
U = VV & OLD & (AMT >= amt_q[:, None])
def prk(x, m): return pd.DataFrame(np.where(m, x, np.nan)).rank(axis=1, pct=True).values
lim = pd.Series(np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=dates) - 0.005
r1 = RC / RC.ffill().shift(1) - 1; one = (RH == RL) & RC.notna()
lockdn = one & r1.le(-lim, axis=0); lockup = one & r1.ge(lim, axis=0); del r1, one
LD, LU = lockdn.values, lockup.values
# ---- 額外指標：EMA200 近 60 日斜率（S60）、60 日波段回撤比（RRT）；皆以個股自身交易日計算
S60 = np.full((T_, N_), np.nan, f32); RRT = np.full((T_, N_), np.nan, f32)
Hn2, Ln2 = H.values, L.values
for j in range(N_):
    ii = np.where(VV[:, j])[0]; n_ = len(ii)
    if n_ < 130: continue
    c_ = Cn[ii, j].astype(float)
    e2 = pd.Series(c_).ewm(span=200, adjust=False).mean().values
    s_ = np.full(n_, np.nan); s_[60:] = e2[60:] / e2[:-60] - 1; S60[ii, j] = s_
    h_ = Hn2[ii, j].astype(float); l_ = Ln2[ii, j].astype(float)
    h_ = np.where(np.isfinite(h_), h_, -np.inf); l_ = np.where(np.isfinite(l_), l_, np.inf)
    tt = np.arange(59, n_); p_ = tt - 59 + swv(h_, 60).argmax(1)        # 近 60 日最高價的位置
    lw = swv(l_, 61).argmin(1); ok = p_ >= 60                              # 高點前 60 日內（含高點）最低價
    q_ = np.where(ok, p_ - 60 + lw[np.where(ok, p_ - 60, 0)], 0)
    pk, lo, cc = h_[p_], l_[q_], c_[tt]
    with np.errstate(all='ignore'):
        rr = (pk - cc) / (pk - lo); sw = pk / lo - 1
    rr = np.where(ok & (sw >= 0.15) & np.isfinite(rr), rr, 0.0)           # 波段漲幅不足 15% 視為無波段，保留
    o_ = np.full(n_, np.nan); o_[tt] = rr; RRT[ii, j] = o_
del Hn2, Ln2; gc.collect(); print('斜率、回撤比完成', flush=True)
# ---- X1
pool = U & (np.nan_to_num(prk(Tr, U)) >= 0.9)
hot = np.nan_to_num(prk(R20, pool)) > 0.8
up5 = pool & (np.nan_to_num(S60, nan=-1.0) >= 0.05)
steep = np.nan_to_num(prk(S60, up5)) > 0.8
ATRP = ATR / Cn
TL = pd.DataFrame(Tr, index=dates)
SCX = 0.5 * np.nan_to_num(prk(Tr, pool), nan=0) + 0.5 * np.nan_to_num(prk((TL - TL.shift(10)).values, pool), nan=0)
def rk_by(sc_, m): return pd.DataFrame(np.where(m, sc_, np.nan)).rank(axis=1, ascending=False, method='first').values
RKX = rk_by(SCX, pool & ~hot & ~steep)
print(f'X1：強勢池平均 {pool.sum(1).mean():.0f} 檔｜剔過熱 {(pool & hot).sum(1).mean():.1f}｜斜率≥5% {up5.sum(1).mean():.1f}｜剔最陡 {(pool & steep).sum(1).mean():.1f}', flush=True)
below = (ix < ix.rolling(200).mean()).values
# ---- V4.1
score = DF(Tr).where(DF(U)); kn = ind.notna().values
g = score.loc[:, kn].T.groupby(ind[kn].values).transform('median').T.reindex(columns=C.columns)
score = score.where((g.where(DF(U)).rank(axis=1, pct=True) >= 0.5) | g.isna()); del g
strong = score.rank(axis=1, pct=True) >= 0.9
buy_ok = ~(ROP.shift(-1) / RC - 1).ge(lim.shift(-1), axis=0) & O.shift(-1).notna()
bear200 = ix < ix.rolling(200).mean(); mv = ix.pct_change().rolling(20).std(); mkt_vol_down = mv < mv.shift(20)
ew = C.pct_change().where(DF(U).shift(1, fill_value=False)).mean(axis=1).fillna(0); ratio_ew = (1 + ew).cumprod() / ix
thrust = (ix.pct_change(10) >= 0.05) & (ratio_ew.pct_change(10) >= 0.02)
thrust_win = thrust.rolling(20, min_periods=1).max().astype(bool); bear_mode = bear200 & ~thrust_win
OFF = iakf_down(ix.dropna()).reindex(dates).fillna(False).rolling(5, min_periods=1).max().astype(bool).values
MD = bear_mode.values
allow = (~MD[:, None] & (atr_ok | thrust_win.values[:, None])) | (MD[:, None] & mkt_vol_down.values[:, None])
deep = np.nan_to_num(RRT, nan=0.0) >= 0.236
sig0 = strong.values & (EV1 | EV2) & allow & ~OFF[:, None] & buy_ok.values
signal = sig0 & ~deep
print(f'V4.1：事件 {sig0.sum()} 筆，剔深回撤 {(sig0 & deep).sum()} 筆（{(sig0 & deep).sum() / max(sig0.sum(), 1):.1%}）', flush=True)
RKV = score.where(DF(signal)).rank(axis=1, ascending=False, method='first').values
del EV1, EV2, atr_ok, allow, signal, sig0, score, strong, buy_ok, S60, RRT; gc.collect()
Ov, Cf = O.values, C.ffill().values
STALE = pd.DataFrame(VV).rolling(20, min_periods=20).sum().values == 0     # 連續 20 個交易日沒有有效價格
del H, L, RH, RL, ROP, V, ATR, NUM, AMT, R20; gc.collect()
print(f'空頭模式 {MD.mean():.1%}｜IAKF 關閉 {OFF.mean():.1%}｜< 200MA {below.mean():.1%}', flush=True)
START = dates[300]; codes = np.array(C.columns)
def sellpx(t, j): return (Ov[t, j] if not np.isnan(Ov[t, j]) else Cf[t, j]) * (1 - SLIP)
# 模擬器：1 倍、隔日開盤成交、漲停買不到、一字跌停賣不掉；回傳日報酬、交易紀錄、持股比例
def _close(tr, j, p, t, px, why):
    tr.append((codes[j], dates[p['t0']], dates[t], p['own'] / p['E'], p['sh'] * px * (1 - FEE - TAX) / p['own'] - 1, t - p['t0'], why, p['st']))
def runX(off=0, seed=0, n=6, drop=0.2, cap=250, ban=None):
    rng = np.random.default_rng(seed); s0 = dates.get_indexer([START], method='bfill')[0] + off
    cash = 1.0; pos = {}; sell, buy = {}, []; eq = np.full(T_, np.nan); ex = np.full(T_, np.nan); tr = []
    for t in range(s0, T_):
        keep = {}
        for j, why in sell.items():
            if j not in pos: continue
            if LD[t, j]: keep[j] = why; continue
            p = pos.pop(j); px = sellpx(t, j); cash += p['sh'] * px * (1 - FEE - TAX); _close(tr, j, p, t, px, why)
        if buy:
            E = cash + sum(p['sh'] * Cf[t - 1, j] for j, p in pos.items())
            for j in buy:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                px *= 1 + SLIP; own = min(E / n, cash)
                if own <= 0: break
                d = ATRP[t - 1, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
                pos[j] = dict(sh=own * (1 - FEE) / px, t0=t, px=px, d=d, hi=px, own=own, E=E, st=False); cash -= own
        sell, buy = keep, []
        mv_ = sum(p['sh'] * Cf[t, j] for j, p in pos.items()); eq[t] = cash + mv_; ex[t] = mv_ / eq[t]
        for j, p in pos.items():
            if STALE[t, j]: p['st'] = True
            c = Cf[t, j]
            if np.isnan(c): continue
            p['hi'] = max(p['hi'], c)
            if c < p['hi'] * (1 - p['d']): sell[j] = '移動停損'
            elif p['hi'] >= p['px'] * 1.06 and c < p['px'] * 1.005: sell[j] = '保本'
            elif t - p['t0'] + 1 >= cap: sell[j] = '250日上限'
        free = n - len(pos) + len(set(sell) & set(pos))
        if free > 0 and not below[t]:
            cnd = np.where(~np.isnan(RKX[t]))[0]
            if ban is not None: cnd = cnd[~np.isin(cnd, ban)]
            if drop: cnd = cnd[rng.random(len(cnd)) >= drop]
            buy = [j for j in cnd[np.argsort(RKX[t, cnd])] if j not in pos and j not in sell][:free]
    for j, p in pos.items(): _close(tr, j, p, T_ - 1, Cf[T_ - 1, j] * (1 - SLIP), '期末未平倉')
    return pd.Series(eq, index=dates).dropna().pct_change().fillna(0), tr, pd.Series(ex, index=dates).dropna()
def runV(off=0, seed=0, n=4, drop=0.2, ban=None):
    rng = np.random.default_rng(seed); s0 = dates.get_indexer([START], method='bfill')[0] + off
    cash = 1.0; pos = {}; sell, buy = {}, []; eq = np.full(T_, np.nan); ex = np.full(T_, np.nan); tr = []
    for t in range(s0, T_):
        keep = {}
        for j, why in sell.items():
            if j not in pos: continue
            if LD[t, j]: keep[j] = why; continue
            p = pos.pop(j); px = sellpx(t, j); cash += p['sh'] * px * (1 - FEE - TAX); _close(tr, j, p, t, px, why)
        if buy:
            E = cash + sum(p['sh'] * Cf[t - 1, j] for j, p in pos.items())
            for j in buy:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                px *= 1 + SLIP; own = min(E / n, cash)
                if own <= 0: break
                pos[j] = dict(sh=own * (1 - FEE) / px, t0=t, px=px, bear=bool(MD[t - 1]), hold=int(rng.integers(18, 23)), own=own, E=E, st=False); cash -= own
        sell, buy = keep, []
        mv_ = sum(p['sh'] * Cf[t, j] for j, p in pos.items()); eq[t] = cash + mv_; ex[t] = mv_ / eq[t]
        for j, p in pos.items():
            if STALE[t, j]: p['st'] = True
            c = Cf[t, j]
            if np.isnan(c): continue
            if OFF[t] and t - p['t0'] >= 1: sell[j] = 'IAKF'
            elif (MD[t] or p['bear']) and c < p['px'] * 0.95: sell[j] = '空頭5%停損'
            elif t - p['t0'] + 1 >= p['hold']: sell[j] = '持有期滿'
        free = n - len(pos) + len(set(sell) & set(pos))
        if free > 0 and not OFF[t]:
            cnd = np.where(~np.isnan(RKV[t]))[0]
            if ban is not None: cnd = cnd[~np.isin(cnd, ban)]
            if drop: cnd = cnd[rng.random(len(cnd)) >= drop]
            buy = [j for j in cnd[np.argsort(RKV[t, cnd])] if j not in pos and j not in sell][:free]
    for j, p in pos.items(): _close(tr, j, p, T_ - 1, Cf[T_ - 1, j] * (1 - SLIP), '期末未平倉')
    return pd.Series(eq, index=dates).dropna().pct_change().fillna(0), tr, pd.Series(ex, index=dates).dropna()
def mix(a, b, wa=0.6):
    a, b = a.align(b, fill_value=0); va, vb, out = wa, 1 - wa, []
    m = a.index.to_period('M')
    for i in range(len(a)):
        if i > 0 and m[i] != m[i - 1]: tot = va + vb; va, vb = tot * wa, tot * (1 - wa)
        prev = va + vb; va *= 1 + a.iloc[i]; vb *= 1 + b.iloc[i]; out.append((va + vb) / prev - 1)
    return pd.Series(out, index=a.index)
def met(r):
    r = r.dropna()
    if len(r) < 60: return (np.nan,) * 4
    e = (1 + r).cumprod(); y = len(r) / 252; cg = e.iloc[-1] ** (1 / y) - 1; md = (e / e.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return cg, md, (cg / -md if md < 0 else np.nan), sh
# ---- 執行
P_ = range(NPATH)
RX = [runX(off=i % 20, seed=1000 + i) for i in P_]; print('X1 完成', flush=True)
RV = [runV(off=i % 20, seed=1000 + i) for i in P_]; print('V4.1 完成', flush=True)
MX = [mix(a[0], b[0]) for a, b in zip(RX, RV)]
TRX = pd.DataFrame([x + (i,) for i, r in enumerate(RX) for x in r[1]], columns=['代碼', '進場', '出場', '權重', '報酬', '持有日', '出場原因', '曾停牌20日', '路徑'])
TRV = pd.DataFrame([x + (i,) for i, r in enumerate(RV) for x in r[1]], columns=TRX.columns[:-1].tolist() + ['路徑'])
TRX['策略'], TRV['策略'] = 'X1', 'V4.1'
TR = pd.concat([TRX, TRV], ignore_index=True)
TR['貢獻'] = TR['權重'] * TR['報酬'] * np.where(TR['策略'] == 'X1', 0.6, 0.4) / NPATH
top = TR.groupby('代碼')['貢獻'].sum().sort_values(ascending=False)
ban = np.where(np.isin(codes, top.index[:5]))[0]
RXb = [runX(off=i % 20, seed=1000 + i, ban=ban)[0] for i in P_]; RVb = [runV(off=i % 20, seed=1000 + i, ban=ban)[0] for i in P_]
MXb = [mix(a, b) for a, b in zip(RXb, RVb)]; print('剔除前 5 名重跑完成', flush=True)
avg = lambda L_: pd.concat(L_, axis=1).fillna(0).mean(axis=1)
CM, CX, CV = avg(MX), avg([r[0] for r in RX]), avg([r[0] for r in RV]); IXR = ix.pct_change().reindex(CM.index).fillna(0)
END = dates[-1]
SEG = ((f'{START:%Y/%m}–2015', START, pd.Timestamp('2015-12-31')), ('2016–2020', pd.Timestamp('2016-01-01'), pd.Timestamp('2020-12-31')),
       (f'2021–{END:%Y/%m}', pd.Timestamp('2021-01-01'), END), (f'2016–{END:%Y/%m}', pd.Timestamp('2016-01-01'), END), ('全期', START, END))
fmt = lambda df, f: df.map(f) if hasattr(df, 'map') else df.applymap(f)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
f1 = lambda x: f'{x:.1%}' if np.isfinite(x) else '—'; f2 = lambda x: f'{x:.2f}' if np.isfinite(x) else '—'
print('\n資料', dates[0].date(), '～', END.date(), '｜回測起點', START.date(), '｜路徑', NPATH, '｜大盤', ix_src)
print('\n=== 1. 分段績效（各路徑中位數；年化 / 最大回撤 / Calmar / Sharpe；另列 Calmar 第 10 百分位與最差路徑回撤）')
rows = []
for nm, P in (('合併 60/40', MX), ('X1 單獨', [r[0] for r in RX]), ('V4.1 單獨', [r[0] for r in RV]), ('合併（剔除前 5 名股票）', MXb)):
    for tag, lo, hi in SEG:
        M = np.array([met(p.loc[lo:hi]) for p in P])
        rows.append(dict(版本=nm, 期間=tag, 年化=f1(np.nanmedian(M[:, 0])), 回撤=f1(np.nanmedian(M[:, 1])), Calmar=f2(np.nanmedian(M[:, 2])),
                         Sharpe=f2(np.nanmedian(M[:, 3])), Calmar_P10=f2(np.nanpercentile(M[:, 2], 10)), 最差回撤=f1(np.nanmin(M[:, 1]))))
for tag, lo, hi in SEG:
    b = met(IXR.loc[lo:hi]); rows.append(dict(版本='加權指數（價格）', 期間=tag, 年化=f1(b[0]), 回撤=f1(b[1]), Calmar=f2(b[2]), Sharpe=f2(b[3])))
print(pd.DataFrame(rows).to_string(index=False))
print('\n=== 2. 每年報酬（路徑平均日報酬複利）')
yr = lambda r: (1 + r).groupby(r.index.year).prod() - 1
Y = pd.DataFrame({'合併': yr(CM), 'X1': yr(CX), 'V4.1': yr(CV), '加權指數': yr(IXR)}); Y['合併−指數'] = Y['合併'] - Y['加權指數']
print(fmt(Y, f1).to_string())
print('\n=== 3. 合併版每月報酬（路徑平均）')
mo = (1 + CM).groupby([CM.index.year, CM.index.month]).prod() - 1; MT = mo.unstack(); MT['全年'] = Y['合併']
print(fmt(MT, lambda x: f'{x * 100:.1f}' if np.isfinite(x) else '').to_string())
print(f'正報酬月份比例 {(mo > 0).mean():.0%}｜最好月 {mo.max():.1%}｜最差月 {mo.min():.1%}')
print('\n=== 4. 合併版前 5 大回撤（路徑平均）')
e = (1 + CM).cumprod(); dd_ = e / e.cummax() - 1; out, i = [], 0
while i < len(dd_):
    if dd_.iloc[i] < 0:
        j = i
        while j < len(dd_) and dd_.iloc[j] < 0: j += 1
        seg = dd_.iloc[i:j]; out.append((e.index[i - 1] if i > 0 else e.index[0], seg.idxmin(), dd_.index[j] if j < len(dd_) else None, seg.min(), j - i)); i = j
    else: i += 1
D = pd.DataFrame(out, columns=['高點', '谷底', '回復', '深度', '天數']).sort_values('深度').head(5)
D['深度'] = D['深度'].map(f1); print(D.to_string(index=False))
print('\n=== 5. 交易統計（' + str(NPATH) + ' 條路徑合計；報酬已扣成本）')
def tstat(d):
    w, l = d['報酬'][d['報酬'] > 0], d['報酬'][d['報酬'] <= 0]; srt = d['報酬'].sort_values(ascending=False)
    gain = d['報酬'].clip(lower=0).sum(); top10 = srt.head(max(1, len(srt) // 10)).clip(lower=0).sum()
    yrs = (END - START).days / 365.25
    return pd.Series({'筆數/年/路徑': f'{len(d) / NPATH / yrs:.1f}', '勝率': f1(len(w) / len(d)), '平均報酬': f1(d['報酬'].mean()), '中位數': f1(d['報酬'].median()),
                      '平均獲利': f1(w.mean()), '平均虧損': f1(l.mean()), '盈虧比': f2(w.mean() / -l.mean()), 'P90': f1(d['報酬'].quantile(0.9)),
                      '前10%交易占總獲利': f1(top10 / gain if gain > 0 else np.nan), '平均持有日': f'{d["持有日"].mean():.0f}', '曾停牌20日以上': int(d['曾停牌20日'].sum())})
print(pd.DataFrame({k: tstat(d) for k, d in TR.groupby('策略')}).to_string())
print('\n出場原因比例'); print(fmt(pd.crosstab(TR['出場原因'], TR['策略'], normalize='columns'), f1).to_string())
print('\n=== 6. 平均持股比例（曝險）')
print(f"X1 {pd.concat([r[2] for r in RX], axis=1).mean(axis=1).mean():.0%}｜V4.1 {pd.concat([r[2] for r in RV], axis=1).mean(axis=1).mean():.0%}")
print('\n=== 7. 對合併版貢獻最大的 10 檔（' + str(NPATH) + ' 路徑平均，單位：初始資金的比例，未複利）')
T10 = TR.groupby('代碼').agg(貢獻=('貢獻', 'sum'), 筆數=('報酬', 'size')).sort_values('貢獻', ascending=False).head(10)
T10['占總貢獻'] = T10['貢獻'] / TR['貢獻'].clip(lower=0).sum(); T10['貢獻'] = T10['貢獻'].map(f1); T10['占總貢獻'] = T10['占總貢獻'].map(f1)
print(T10.to_string()); print('第 1 節「合併（剔除前 5 名股票）」即排除上表前 5 檔後重跑的結果。')
pd.DataFrame({'合併': CM, 'X1': CX, 'V4.1': CV, '加權指數': IXR}).to_csv('small_merge_daily.csv')
TR.to_csv('small_merge_trades.csv', index=False); print('\n已存檔：small_merge_daily.csv、small_merge_trades.csv')
