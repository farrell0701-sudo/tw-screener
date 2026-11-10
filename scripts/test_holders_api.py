"""一次性測試:FinMind TaiwanStockHoldingSharesPer 是否能下載(不印出 token)。"""
import os
import requests

TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
H = {"Authorization": f"Bearer {TOKEN}"}
print("token 是否存在:", bool(TOKEN), "長度:", len(TOKEN))

try:
    r = requests.get("https://api.web.finmindtrade.com/v2/user_info", headers=H, timeout=60)
    j = r.json()
    print("user_info HTTP", r.status_code, "level:", j.get("level"), "level_title:", j.get("level_title"),
          "api_request_limit:", j.get("api_request_limit"), "user_count:", j.get("user_count"),
          "msg:", str(j.get("msg"))[:150])
except Exception as e:  # noqa: BLE001
    print("user_info 失敗", e)

CASES = [
    ("2330 短區間 4 週", dict(data_id="2330", start_date="2024-01-01", end_date="2024-01-31")),
    ("2330 一年", dict(data_id="2330", start_date="2023-01-01", end_date="2023-12-31")),
    ("2330 2015 起全部", dict(data_id="2330", start_date="2015-01-01")),
    ("2317 2010 起全部", dict(data_id="2317", start_date="2010-01-01")),
    ("全市場單一週(不給代號)", dict(start_date="2024-01-05", end_date="2024-01-05")),
    ("對照:2330 日K(免費資料集)", dict(data_id="2330", start_date="2024-01-01", end_date="2024-01-10")),
]
for name, p in CASES:
    ds = "TaiwanStockPrice" if "日K" in name else "TaiwanStockHoldingSharesPer"
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={"dataset": ds, **p}, headers=H, timeout=120)
        j = r.json()
        d = j.get("data") or []
        dates = sorted({x.get("date") for x in d})
        print(f"[{name}] HTTP {r.status_code} status={j.get('status')} rows={len(d)} 週數={len(dates)} "
              f"期間={dates[0] if dates else '-'}~{dates[-1] if dates else '-'} msg={str(j.get('msg'))[:160]}")
        if d and ds != "TaiwanStockPrice":
            print("   欄位:", list(d[0].keys()))
            print("   範例:", d[:2])
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] 失敗 {e}")
