# 籌碼雷達(台股大戶籌碼 × 技術型態篩選網站)

靜態網站 + GitHub Actions 自動更新。不需要伺服器,免費部署在 GitHub Pages。

- 集保「股權分散表」:每週六自動抓,計算 400 張 / 1,000 張以上大戶比例、連續增加週數、N 週變化
- 日K 股價:每個交易日 15:40 自動抓(yfinance,還原權息)
- 技術型態:A→B 突破 v2、放量推升後收斂突破、波動收斂比、EMA150 斜率、距 120 日高
- 網頁篩選條件可自由組合,點股票名稱會開 TradingView 圖表

## 一次性設定(約 10 分鐘)

1. 在 GitHub 建立一個新的 repository(例如 `tw-screener`),Public 或 Private 皆可
   (Private repo 使用 GitHub Pages 需要付費方案;免費帳號請用 Public)。
2. 把這個資料夾的所有檔案上傳到 repo 根目錄(包含隱藏資料夾 `.github`)。
   最簡單的方法:repo 頁面 → Add file → Upload files → 把資料夾內容整個拖進去。
   注意 `.github/workflows/update.yml` 一定要在。
3. repo → Settings → Pages → Build and deployment → Source 選 **GitHub Actions**。
4. repo → Settings → Actions → General → Workflow permissions 選 **Read and write permissions**,存檔。
5. repo → Actions → 「更新資料並發布網站」→ Run workflow,手動跑第一次。
   第一次要下載約 1,900 檔兩年日K,大約 10~20 分鐘。
6. 跑完後網址會出現在 Actions 執行結果與 Settings → Pages,格式為
   `https://<你的帳號>.github.io/<repo 名稱>/`。

## 即時報價(盤中)

網頁右上的「即時報價」開關會向 Cloudflare Worker 查詢目前列表前 40 檔的最新價(約每分鐘一次,僅盤中 09:00~13:35)。
Worker 程式在 `worker/worker.js`,富果 API 金鑰只存在 Worker 的 Secret(`FUGLE_KEY`),不會出現在網頁或 GitHub。

- `site/index.html` 的 script 開頭有 `QUOTE_URL`,要填你的 Worker 網址;留空字串可關閉此功能。
- 富果免費版每分鐘 60 次,所以每次最多 40 檔,且 Worker 會快取 45 秒。要全市場即時請升級付費方案。
- 集保大戶比例每週才公布一次,不會即時變動;A→B 等型態仍以收盤資料判斷,盤中只有「盤中創120日高」是用即時價判斷。
- 建議在 Cloudflare 把 Worker 的變數 `ALLOW_ORIGIN` 設成你的網站網址(例如 `https://帳號.github.io`),避免別人拿你的額度。
- 國定假日休市時程式不知道,會當成盤中查詢,只是價格不會變動。

## 重要限制

- **集保歷史從第一次執行開始累積。** 集保開放資料只提供最新一週,所以「連續增加週數」、
  「近 13 週」要等排程跑幾週後才完整(第一週只有 1 週資料)。
- yfinance 是非官方的 Yahoo 資料介面,偶爾會失敗或缺資料;失敗時隔天排程會自動補上。
- 市值以「集保總股數 × 收盤價」估算,與官方市值可能略有差異。
- GitHub 排程可能延遲數分鐘到數十分鐘,屬正常現象。

## 檔案結構

```
.github/workflows/update.yml   排程:抓資料 → 計算 → 提交 → 發布網站
scripts/fetch_list.py          上市上櫃普通股清單(證交所 ISIN 頁)
scripts/fetch_tdcc.py          集保股權分散表(每週一份,存 data/tdcc/)
scripts/fetch_prices.py        日K(存 data/prices/,每檔一個 CSV)
scripts/build.py               計算籌碼與技術指標 → site/data/screen.json
site/index.html                篩選網頁(單一檔案)
worker/worker.js               Cloudflare Worker(即時報價中繼,貼到 Cloudflare)
```

## 調整型態參數

`scripts/build.py` 最上方的 `AB_P`(A→B 突破)與 `VCP_P`(收斂突破)可以改,
改完 push 後在 Actions 手動執行一次即可。

## 本機測試

```
pip install -r requirements.txt
python scripts/fetch_list.py && python scripts/fetch_tdcc.py && python scripts/fetch_prices.py && python scripts/build.py
cd site && python -m http.server 8000   # 瀏覽器開 http://localhost:8000
```

資料僅供研究參考,不構成投資建議。
