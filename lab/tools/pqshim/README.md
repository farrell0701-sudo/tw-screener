# pqshim：沒有 pyarrow 的環境讀 parquet

雲端沙箱沒有 pyarrow、也不能 pip 安裝時使用。純 Python 讀 parquet（ZSTD 透過系統 libzstd），`pd.read_parquet` 會被替換。

用法（在 tw-screener 根目錄）：

```
PYTHONPATH=lab/tools/pqshim python3 lab/core/run.py <編號> --no-push
```

`sitecustomize.py` 讓 multiprocessing 子程序也自動載入。有 pyarrow 的環境不需要。
