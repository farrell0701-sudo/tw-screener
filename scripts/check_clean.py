import glob, os, sys
import numpy as np, pandas as pd
D = sys.argv[1]; rows = []
for p in glob.glob(f"{D}/[0-9]*.csv"):
    d = pd.read_csv(p, parse_dates=["date"])
    if len(d) < 7: continue
    ac = d.c * d.af; r = ac.pct_change()
    gapd = d.date.diff().dt.days
    lim = np.where(d.date >= "2015-06-01", 0.105, 0.075)
    ok = (np.arange(len(d)) >= 5) & (gapd <= 10)        # 排除上市前 5 天與停牌後恢復
    bad = ok & (r.abs() > lim)
    for i in np.where(bad)[0]:
        rows.append((os.path.basename(p)[:-4], d.date[i], r[i]))
    if (ac <= 0).any(): rows.append((os.path.basename(p)[:-4], None, -999))
b = pd.DataFrame(rows, columns=["code", "date", "r"])
print("超過漲跌停的筆數:", len(b), " 負價:", (b.r == -999).sum())
b = b.dropna(); y = b.date.dt.year
print(b.groupby(y).size().to_string())
print("跌", (b.r < 0).sum(), "漲", (b.r > 0).sum())
b.to_csv(f"{D}/_check.csv", index=False)
