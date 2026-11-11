"""外部市場資料（回測／急殺預警研究用，不影響網站）。
來源：FinMind API v4（GitHub Secret FINMIND_TOKEN，可無）＋ Yahoo Finance（yfinance）。
輸出到 out_ext/，由 workflow 推送到分支 ext-data。"""
import os, sys, time
from pathlib import Path
import pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_ext"; OUT.mkdir(exist_ok=True)
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2 if TOKEN else 12.5
START, END = 2004, pd.Timestamp.today().year
_last = [0.0]; LOG = []

def log(s):
    print(s, flush=True); LOG.append(s)

def fm(dataset, data_id=None, y0=START):
    frames = []
    for y in range(y0, END + 1):
        p = {"dataset": dataset, "start_date": f"{y}-01-01", "end_date": f"{y}-12-31"}
        if data_id: p["data_id"] = data_id
        for _ in range(5):
            w = GAP - (time.time() - _last[0])
            if w > 0: time.sleep(w)
            _last[0] = time.time()
            try:
                h = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
                r = requests.get(URL, params=p, headers=h, timeout=180); j = r.json()
            except Exception as e:  # noqa: BLE001
                log(f"  {dataset} {y} 連線失敗 {e}"); time.sleep(20); continue
            if r.status_code == 200 and j.get("status") == 200:
                d = pd.DataFrame(j.get("data", []))
                if len(d): frames.append(d)
                break
            msg = str(j.get("msg", ""))[:160]
            if r.status_code == 429 or "limit" in msg.lower() or "upper" in msg.lower():
                log(f"  達查詢上限，暫停 10 分鐘：{msg}"); time.sleep(600); continue
            log(f"  {dataset} {data_id or ''} {y} 回應 {r.status_code} {msg}"); break
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    name = dataset + (f"_{data_id}" if data_id else "")
    if len(df):
        df.to_csv(OUT / f"fm_{name}.csv", index=False, encoding="utf-8-sig")
        log(f"[OK] {name} {df.shape} {df['date'].min() if 'date' in df else ''}~{df['date'].max() if 'date' in df else ''} 欄位 {list(df.columns)[:15]}")
    else:
        log(f"[空] {name}")

def yahoo():
    try:
        import yfinance as yf
    except Exception as e:  # noqa: BLE001
        log(f"[yfinance 無法載入] {e}"); return
    syms = ["^GSPC", "^IXIC", "^SOX", "^VIX", "^N225", "^TWII", "^KS11", "^HSI", "TWD=X", "JPY=X", "DX-Y.NYB", "^TNX", "HYG", "TLT"]
    out = {}
    for s in syms:
        try:
            d = yf.download(s, start="2003-01-01", progress=False, auto_adjust=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            if len(d):
                for c in ("Close", "High", "Low"):
                    out[f"{s}_{c}"] = d[c]
                log(f"[OK] yahoo {s} {len(d)} {d.index.min().date()}~{d.index.max().date()}")
            else:
                log(f"[空] yahoo {s}")
        except Exception as e:  # noqa: BLE001
            log(f"[失敗] yahoo {s} {e}")
    if out:
        pd.DataFrame(out).sort_index().to_csv(OUT / "yahoo.csv", encoding="utf-8-sig")

def disposal():
    """證交所處置公告（rwd API），按季抓；櫃買中心嘗試兩個端點，失敗只記錄。"""
    frames = []
    for y in range(2012, END + 1):
        for q0, q1 in (("0101", "0331"), ("0401", "0630"), ("0701", "0930"), ("1001", "1231")):
            u = f"https://www.twse.com.tw/rwd/zh/announcement/punish?startDate={y}{q0}&endDate={y}{q1}&response=json"
            try:
                j = requests.get(u, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).json()
                rows, fields = j.get("data") or [], j.get("fields") or []
                if rows: frames.append(pd.DataFrame(rows, columns=fields[:len(rows[0])]).assign(market="TWSE"))
            except Exception as e:  # noqa: BLE001
                log(f"  TWSE 處置 {y}{q0} 失敗 {e}")
            time.sleep(3)
    for u in ["https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?response=json&startDate=2012/01/01&endDate=" + pd.Timestamp.today().strftime("%Y/%m/%d"),
              "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information"]:
        try:
            r = requests.get(u, timeout=60, headers={"User-Agent": "Mozilla/5.0"}); j = r.json()
            tb = j.get("tables", [{}])[0] if isinstance(j, dict) and "tables" in j else None
            if tb and tb.get("data"):
                frames.append(pd.DataFrame(tb["data"], columns=tb.get("fields")).assign(market="TPEx")); log(f"[OK] TPEx 處置 {u[:60]} {len(tb['data'])}")
            elif isinstance(j, list) and j:
                frames.append(pd.DataFrame(j).assign(market="TPEx")); log(f"[OK] TPEx 處置 openapi {len(j)}（只有近期）")
            else:
                log(f"[空] TPEx 處置 {u[:60]} 回應 {r.status_code} {str(j)[:120]}")
        except Exception as e:  # noqa: BLE001
            log(f"[失敗] TPEx 處置 {u[:60]} {e}")
    if frames:
        d = pd.concat(frames, ignore_index=True); d.to_csv(OUT / "disposal.csv", index=False, encoding="utf-8-sig")
        log(f"[OK] 處置股 {d.shape} 欄位 {list(d.columns)[:12]}")
    else:
        log("[空] 處置股")

if __name__ == "__main__":
    log(f"token: {'有' if TOKEN else '無'}")
    yahoo()
    fm("TaiwanStockTotalMarginPurchaseShortSale")
    fm("TaiwanStockTotalInstitutionalInvestors")
    fm("TaiwanFuturesInstitutionalInvestors", "TX")
    fm("TaiwanFuturesInstitutionalInvestors", "MTX")
    fm("TaiwanOptionInstitutionalInvestors", "TXO")
    fm("TaiwanExchangeRate", "USD")
    fm("TaiwanExchangeRate", "JPY")
    disposal()   # 處置股（證交所／櫃買中心公告；FinMind 此資料需付費等級）
    (OUT / "README.md").write_text("外部市場資料（急殺預警研究用）\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
