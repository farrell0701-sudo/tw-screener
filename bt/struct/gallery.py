import os, sys
os.environ['RAWSRC']='finmind'; os.environ['DELIST']='1'; sys.path.insert(0,'/home/claude/bt')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp=[f for f in fm.findSystemFonts() if 'notosanscjk' in f.lower()]
if fp: fm.fontManager.addfont(fp[0]); plt.rcParams['font.family']=fm.FontProperties(fname=fp[0]).get_name()
plt.rcParams['axes.unicode_minus']=False
from load import load_all
lst,tx,cal,S,_=load_all(); names=lst.set_index('code').name
E=pd.read_pickle('/home/claude/bt/struct/events.pkl'); E=E[E.complete]
smp=E.sample(20,random_state=7).sort_values('date')
bg,fg='#111418','#d8dee6'; colmap={'win':'#e5484d','loss':'#2bb673','none':'#9aa4b2'}; labzh={'win':'成功','loss':'失敗','none':'未定'}
fig,axs=plt.subplots(5,4,figsize=(20,22),dpi=80); fig.patch.set_facecolor(bg)
for ax,(_,e) in zip(axs.flat,smp.iterrows()):
    a=S[e.code]; t=a.index.get_loc(e.date); w=a.iloc[max(0,t-140):t+61]; x=np.arange(len(w)); tt=t-max(0,t-140)
    up=w.C>=w.C.shift(1)
    ax.set_facecolor(bg); ax.vlines(x,w.L,w.H,color=np.where(up,'#e5484d','#2bb673'),lw=0.7)
    ax.plot(x,w.C.ewm(span=12,adjust=False).mean(),color='white',lw=0.7)
    ax.axhline(e.c_top,color='#f5a524',ls='--',lw=0.9)
    ax.axvspan(max(0,tt-120),tt-20,color='#f5a524',alpha=0.08)
    ax.axvspan(tt-4,tt+0.5,color='#c084fc',alpha=0.35)
    ax.axvline(tt,color='#c084fc',lw=0.8)
    ax.set_title(f"{e.code} {names.get(e.code,'')}  {e.date:%Y-%m-%d}  {labzh[e.label]}  60日 {e.fwd60:+.0%}",
                 color=colmap[e.label],fontsize=11,loc='left')
    ax.set_xticks([]); ax.tick_params(colors=fg,labelsize=8); [s.set_color('#333a44') for s in ax.spines.values()]
fig.suptitle('隨機 20 個結構轉變事件（橘底＝遠期比較區、橘虛線＝C 上緣、紫底＝A 區、紫線＝判定日；右側 60 日為事後走勢）',color=fg,fontsize=15,y=0.995)
plt.tight_layout(rect=(0,0,1,0.985)); plt.savefig('/home/claude/bt/struct/gallery20.png',facecolor=bg)
print(smp[['code','date','label','fwd60','ovh_vol_12m','dec_drop20','lo_ratio']].round(3).to_string())
