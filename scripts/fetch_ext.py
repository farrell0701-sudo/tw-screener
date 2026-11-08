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
    (OUT / "README.md").write_text("外部市場資料（急殺預警研究用）\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
