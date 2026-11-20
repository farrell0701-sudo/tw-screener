# 沙盒專用：每個 Python 程序（含 multiprocessing spawn 的子程序）啟動時自動裝上 parquet 替代讀取器
try:
    import pqshim; pqshim.install()
except Exception as e:
    import sys; print('pqshim 安裝失敗', e, file=sys.stderr)
