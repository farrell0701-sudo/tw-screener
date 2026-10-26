"""次強股點火：T_ldf 前 10～30% 的收斂突破事件"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sleeve.py').read().split("\nif __name__")[0])
pct = cur.rank(axis=1, pct=True)
pools = {'前 10%（現行）': strong, '前 10～20%': (pct >= 0.8) & (pct < 0.9) & ~ldn5, '前 20～30%': (pct >= 0.7) & (pct < 0.8) & ~ldn5,
         '前 30～50%': (pct >= 0.5) & (pct < 0.7) & ~ldn5}
ent = O.shift(-1); Cf = C.ffill(); dd0 = pd.Timestamp('2016-02-01'); lu_next = (O.shift(-1) / Cf - 1 >= 0.095)
rows = []
for lab, pool in pools.items():
    base_m = (pool & ~lu_next & ent.notna()).loc[dd0:]; evm = base_m & anyev.loc[dd0:]
    bstrong = (strong & ~lu_next & ent.notna()).loc[dd0:]
    o = {'股票群': lab, '每年事件': int(evm.sum().sum() / 10.7)}
    for h in (5, 20, 60):
        Rr = (Cf.shift(-h) / ent - 1).loc[dd0:]; m = evm & Rr.notna()
        ex_own = Rr.sub(Rr.where(base_m).mean(1), axis=0).where(m).stack().dropna()       # 相對同群
        ex_st = Rr.sub(Rr.where(bstrong).mean(1), axis=0).where(m).stack().dropna()       # 相對前 10% 強勢股
        yr = ex_own.groupby(ex_own.index.get_level_values(0).year).mean()
        o[f'{h}日 相對同群'] = f'{ex_own.mean():+.2%}'; o[f'{h}日 正年'] = f'{(yr > 0).sum()}/{len(yr)}'; o[f'{h}日 相對強勢股'] = f'{ex_st.mean():+.2%}'
    rows.append(o)
pd.set_option('display.width', 260); print(pd.DataFrame(rows).to_string(index=False), flush=True)
sleeve(anyev, '基準：前 10% 的收斂突破')
sleeve(anyev, '只用前 10～20%', pool=pools['前 10～20%'])
sleeve(anyev, '只用前 10～30%', pool=pools['前 10～20%'] | pools['前 20～30%'])
sleeve(anyev, '擴大到前 20%，依 T_ldf 排序', pool=pools['前 10%（現行）'] | pools['前 10～20%'])
sleeve(anyev, '擴大到前 30%，依 T_ldf 排序', pool=pools['前 10%（現行）'] | pools['前 10～20%'] | pools['前 20～30%'])
sleeve(anyev, '擴大到前 30%，15 個部位', pool=pools['前 10%（現行）'] | pools['前 10～20%'] | pools['前 20～30%'], n=15)
sleeve(anyev, '前 10%，15 個部位（對照）', n=15)
print(pd.DataFrame(RES).to_string(index=False))
