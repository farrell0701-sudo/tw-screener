"""抓取集保結算所「集保戶股權分散表」最新一週資料,存成 data/tdcc/YYYYMMDD.csv。

資料來源:集保結算所開放資料 https://opendata.tdcc.com.tw/getOD.ashx?id=1-5
欄位:資料日期, 證券代號, 持股分級, 人數, 股數, 占集保庫存數比例%
持股分級:1~15 為各級距,16 為差異數調整,17 為合計。
  12 = 400,001~600,000 股,13 = 600,001~800,000,14 = 800,001~1,000,000,15 = 1,000,001 股以上

注意:開放資料只提供「最新一週」,歷史會隨每週排程自動累積。
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "tdcc"
URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
HEADERS = {"User-Agent": "Mozilla/5.0 (tw-screener)"}


def main() -> None:
    r = requests.get(URL, headers=HEADERS, timeout=120)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = ["date", "code", "level", "holders", "shares", "pct"][: len(df.columns)]
    df["code"] = df["code"].str.strip()
    df = df[df["code"].str.fullmatch(r"\d{4}")]
    date = df["date"].dropna().iloc[0].strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{date}.csv"
    if out.exists():
        print(f"[fetch_tdcc] {date} 已存在,略過")
        return
    df[["code", "level", "holders", "shares", "pct"]].to_csv(out, index=False)
    print(f"[fetch_tdcc] {date}: {df['code'].nunique()} 檔 → {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[fetch_tdcc] 失敗: {e}", file=sys.stderr)
        sys.exit(1)
