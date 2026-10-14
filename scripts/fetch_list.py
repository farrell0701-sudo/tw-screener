"""抓取上市、上櫃普通股清單(代號、名稱、市場、產業別),存成 data/stock_list.csv。

資料來源:臺灣證券交易所 ISIN 查詢頁 https://isin.twse.com.tw
  strMode=2 → 上市,strMode=4 → 上櫃
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stock_list.csv"
URLS = {
    "TWSE": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
    "TPEX": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (tw-screener)"}


def fetch(market: str, url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.encoding = "big5"  # 頁面為 Big5 / MS950
    tables = pd.read_html(io.StringIO(r.text), header=0)
    df = tables[0]
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]  # 有價證券代號及名稱
    df = df[df[first].astype(str).str.contains("　")]  # 代號與名稱以全形空白分隔
    parts = df[first].astype(str).str.split("　", n=1, expand=True)
    out = pd.DataFrame({
        "code": parts[0].str.strip(),
        "name": parts[1].str.strip(),
        "market": market,
        "industry": df.get("產業別", pd.Series([""] * len(df))).fillna("").astype(str).values,
        "cfi": df.get("CFICode", pd.Series([""] * len(df))).fillna("").astype(str).values,
    })
    # 只留 4 碼普通股(CFICode 以 ES 開頭)
    out = out[out["code"].str.fullmatch(r"\d{4}") & out["cfi"].str.startswith("ES")]
    return out.drop(columns=["cfi"])


def main() -> None:
    frames = []
    for m, u in URLS.items():
        try:
            frames.append(fetch(m, u))
        except Exception as e:  # noqa: BLE001
            print(f"[fetch_list] {m} 失敗: {e}", file=sys.stderr)
    if not frames:
        if OUT.exists():
            print("[fetch_list] 兩個市場都失敗,沿用舊清單")
            return
        sys.exit(1)
    df = pd.concat(frames).drop_duplicates("code").sort_values("code")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[fetch_list] {len(df)} 檔 → {OUT}")


if __name__ == "__main__":
    main()
