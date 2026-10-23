import pandas as pd, numpy as np
exec(open('test_ema_mock.py').read().split("M.TRAIN=")[0])
M.TRAIN=('2016-06-01','2020-12-31'); M.TEST=('2021-01-01','2099-12-31'); M.N_STARTS=1
M.run_all(P, n_random=2)
