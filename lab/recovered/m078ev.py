import pandas as pd, numpy as np
E=pd.concat([pd.read_pickle(f'/home/claude/old/M078_{w}.pkl')['E'] for w in ['cpold','cpnew']]); B=pd.concat([pd.read_pickle(f'/home/claude/old/M078_{w}.pkl')['B'] for w in ['cpold','cpnew']])
for X in (E,B): X['seg']=np.where(X.date.dt.year<=2015,'05-15',np.where(X.date.dt.year<=2020,'16-20','21-26'))
S=['05-15','16-20','21-26']; EX=['F20','F40','F60','TR','F20S','F40S','F60S','TRS']; bas={'F20S':'F20','F40S':'F40','F60S':'F60','TRS':'TR'}
N0=E[~E.anyp]; P=E[E.anyp]
print('件數（型態事件／無型態觸底／池隨機）:', {s:(int((P.seg==s).sum()),int((N0.seg==s).sum()),int((B.seg==s).sum())) for s in S})
print('型態件數:', {k:[int(P[P.seg==s][k].sum()) for s in S] for k in ['ENG','GAP','SPR','VREC','HAM']}, '跌破占比', [round(P[P.seg==s].brk.mean(),2) for s in S])
print('\n== 對照 A 池隨機（平均超額%／中位）'); 
for x in ['F20','F40','F60','TR']: print(f' {x:4s} '+' ｜ '.join(f'{B[B.seg==s][x].mean()*100:+.2f}/{B[B.seg==s][x].median()*100:+.2f}' for s in S))
print('== 對照 B 二次觸底無型態')
for x in EX: print(f' {x:4s} '+' ｜ '.join(f'{N0[N0.seg==s][x].mean()*100:+.2f}/{N0[N0.seg==s][x].median()*100:+.2f}' for s in S))
print('\n== 型態 × 出場：平均超額／中位／勝率／n ｜ 減A ｜ 減B   （* = 三段全過判準）')
groups={'ANY':P,'ANY觸碰':P[~P.brk],'ANY跌破':P[P.brk],'ANY且當日在池':P[P.pool_t]}; groups.update({k:P[P[k]] for k in ['ENG','GAP','SPR','VREC','HAM']})
for g,G in groups.items():
    print(f'-- {g}')
    for x in EX:
        line=[]; ok=True
        for s in S:
            a=G[G.seg==s][x].dropna(); bA=B[B.seg==s][bas.get(x,x)].mean(); bB=N0[N0.seg==s][x].mean()
            if len(a)==0: ok=False; line.append('n0'); continue
            dA=a.mean()-bA; dB=a.mean()-bB; ok&=(len(a)>=100 and dA>0.0077 and dB>0 and a.median()>0)
            line.append(f'{a.mean()*100:+.2f}/{a.median()*100:+.2f}/{(a>0).mean()*100:.0f}%/n{len(a)} ｜{dA*100:+.2f}｜{dB*100:+.2f}')
        print(f' {x:4s}{"*" if ok else " "} '+'  ‖  '.join(line))
print('\n== 硬停損觸發率、移動停損平均持有日')
for s in S: G=P[P.seg==s]; print(f' {s} 觸發 {G.stopped.mean()*100:.0f}% ｜ 移動停損持有 {G.TRd.mean():.0f} 日 ｜ 被停損者若續抱 F20 {G[G.stopped].F20.mean()*100:+.2f} 未停損者 {G[~G.stopped].F20.mean()*100:+.2f}')
