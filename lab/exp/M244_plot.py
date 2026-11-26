import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
d = 'lab/results/M244/'; Z = np.load(d + 'Z_avg.npy'); J = json.load(open(d + 'tree.json'))
ok = {'R6': '✓ 確認增量', 'POS250': '✓ 確認增量', 'DH250': '✓ 確認增量（最強）', 'S': '基準', 'B4': '✓ 確認（反向）'}
labs = [f"{f}  {J['lab'][f]}" + (f"   {ok[f]}" if f in ok else '') for f in J['fac']]
fig, ax = plt.subplots(figsize=(9.5, 9.6), dpi=160)
ink, mute = '#1f2328', '#8a8f98'
dn = dendrogram(Z, orientation='right', labels=labs, color_threshold=0, above_threshold_color=mute, ax=ax, leaf_font_size=10)
for c in dn['icoord']: pass
for t in ax.get_yticklabels():
    s = t.get_text(); t.set_color(ink)
    if '✓' in s: t.set_fontweight('bold')
ax.axvline(0.5, color=ink, lw=1, ls=(0, (4, 3)))
ax.text(0.495, 2, '切點 |ρ| = 0.5', color=ink, fontsize=9, va='center', ha='left', backgroundcolor='white')
ax.set_xlim(1.0, 0)
ax.set_xticks([1, 0.8, 0.6, 0.4, 0.2, 0]); ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=mute)
ax.set_xlabel('合併處的相關強度 |ρ|（越往右越像同一件事）', color=ink, fontsize=10)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(mute); ax.tick_params(axis='y', length=0)
fig.suptitle('X1 候選內 26 個因子的相關樹（含籌碼）', fontsize=13, color=ink, x=0.02, ha='left', y=0.985)
fig.text(0.02, 0.928, f"奇數代碼；2010–15 與 2016–24 平均（集保資料自 2010 起）；兩期樹形幾乎相同（穩定度 {J['stab']:.2f}）", fontsize=9.5, color=mute, ha='left')
plt.tight_layout(rect=(0, 0, 1, 0.91)); plt.savefig(d + 'M244_tree.png', facecolor='white')
