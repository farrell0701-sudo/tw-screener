"""一次性探測：FinMind 財報與本益比資料最早到哪一年（2330、1101、2317；只看筆數與日期範圍）。"""
import os, time, requests
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
H = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
out = [f"token: {'有' if TOKEN else '無'}"]
for ds in ["TaiwanStockFinancialStatements", "TaiwanStockBalanceSheet", "TaiwanStockPER", "TaiwanStockCashFlowsStatement"]:
    for code in ["2330", "1101", "2317"]:
        try:
            r = requests.get(URL, params={"dataset": ds, "data_id": code, "start_date": "2000-01-01", "end_date": "2014-12-31"}, headers=H, timeout=120)
            j = r.json(); d = j.get("data", [])
            ds_ = sorted({x["date"] for x in d})
            out.append(f"{ds} {code} status {j.get('status')} msg {j.get('msg')} 筆數 {len(d)} 最早 {ds_[0] if ds_ else '-'} 最晚(≤2014) {ds_[-1] if ds_ else '-'}")
        except Exception as e:  # noqa: BLE001
            out.append(f"{ds} {code} 失敗 {e}")
        time.sleep(7)
os.makedirs("out_probe", exist_ok=True)
open("out_probe/result.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
