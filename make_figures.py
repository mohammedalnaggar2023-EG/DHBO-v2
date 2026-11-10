"""Draws the figures of the paper and Supplementary Figure S1.

Output (folder figures/, PNG and TIFF at 1200 dpi):
  Fig1  structure of DHBO-v2          Fig2  Diebold-Mariano statistics (Table S6)
  Fig3  interval coverage (Table S7)  Fig4  relative MASE, development vs held-out
  Fig5  reporting regimes, Brazil and USA
  FigS1 causal VMD of the India window ending 30 December 2021
Usage:  python make_figures.py      (needs data/confirmed_global.csv and vmdpy)
"""
import os, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from vmdpy import VMD
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_data import dev, held
from matplotlib import font_manager
_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ('Liberation Sans', 'Arial', 'Helvetica') if f in _have), 'DejaVu Sans')
MM=1/25.4; FULL=174*MM; COL=84*MM
plt.rcParams.update({'font.family':FONT,'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':8,
 'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'mathtext.fontset':'custom','mathtext.rm':FONT,
 'mathtext.it':FONT+':italic','mathtext.bf':FONT+':bold','mathtext.sf':FONT,'mathtext.cal':FONT+':italic','savefig.dpi':1200})
def save(fig,name):
    fig.savefig(f'{OUTDIR}/{name}.png',dpi=1200)
    fig.savefig(f'{OUTDIR}/{name}.tif',dpi=1200,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
OUTDIR=os.environ.get('DHBO_FIGS','figures'); os.makedirs(OUTDIR,exist_ok=True)
INK='#22313f'; BLUE='#e3edf7'; GREEN='#e6f1e4'; SAND='#fbf0dc'; GREY='#f1f1f1'
# ---------------- data
df=pd.read_csv(os.environ.get('DHBO_DATA', 'data/confirmed_global.csv'))
nat=df.drop(columns=['Province/State','Lat','Long']).groupby('Country/Region').sum().T
nat.index=pd.to_datetime(nat.index,format='%m/%d/%y')
cum=nat.cummax(); daily=cum.diff().clip(lower=0); ma=daily.rolling(7).mean()

# ---------------- Figure 1
def box(ax,x,y,w,h,txt,fc,dashed=False,bold=False,fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec=INK,lw=0.9,ls=(0,(3.5,2)) if dashed else '-'))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=INK,weight='bold' if bold else 'normal',linespacing=1.2)
def arr(ax,x1,y1,x2,y2,c=INK,ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=c,lw=0.8,ls=ls,shrinkA=0,shrinkB=0,mutation_scale=7))
fig=plt.figure(figsize=(FULL,FULL*0.50)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,22.4); ax.set_ylim(0,11.2); ax.axis('off')
ax.plot([5.2,6.0],[10.8,10.8],color=INK,lw=0.9,ls=(0,(3.5,2))); ax.text(6.2,10.8,'dashed border = tuned by Bayesian optimisation (TPE)',va='center',fontsize=8,color=INK)
box(ax,0.1,4.8,2.3,2.0,'Log series\n$z_1,\\ldots,z_t$',GREY)
box(ax,2.95,8.3,3.4,1.7,'SARIMA',BLUE,bold=True)
box(ax,2.95,4.7,3.4,2.2,'Causal VMD\n(window $w_t$)',BLUE,dashed=True)
box(ax,2.95,2.3,3.4,1.3,'Damped ETS',BLUE)
box(ax,2.95,0.3,3.4,1.3,'Naive',BLUE)
box(ax,6.9,4.3,2.75,3.1,'Inputs:\nlevel,\ninnovations,\nVMD modes',GREY)
box(ax,10.15,7.0,3.15,1.8,'BiLSTM +\nattention',GREEN,dashed=True)
box(ax,10.15,3.2,3.15,1.8,'TCN',GREEN,dashed=True)
box(ax,13.8,7.0,2.4,1.8,'Hybrid A\n$\\hat{s}+\\hat{r}^{A}$',GREEN)
box(ax,13.8,3.2,2.4,1.8,'Hybrid B\n$\\hat{s}+\\hat{r}^{B}$',GREEN)
box(ax,16.7,0.3,2.55,9.7,'Online\nconvex\nstacking\n\n5 inputs\n60-origin\nwindow',SAND,bold=True)
box(ax,19.65,6.2,2.65,2.3,'Adaptive\nconformal\nintervals',SAND,bold=True)
box(ax,19.65,2.6,2.65,2.6,'Forecast\n$t+1,\\ldots,t+7$\n80% / 95%',GREY)
arr(ax,2.4,6.2,2.95,8.7); arr(ax,2.4,5.8,2.95,5.8); arr(ax,2.4,5.3,2.95,2.95); arr(ax,2.4,5.0,2.95,0.95)
arr(ax,6.35,8.6,7.6,7.4); arr(ax,6.35,5.8,6.9,5.8)
arr(ax,9.65,6.8,10.15,7.9); arr(ax,9.65,4.9,10.15,4.1)
arr(ax,13.3,7.9,13.8,7.9); arr(ax,13.3,4.1,13.8,4.1)
arr(ax,6.35,9.55,16.7,9.55); ax.text(11.5,9.85,'SARIMA forecast $\\hat{s}$ (base of both hybrids)',ha='center',fontsize=8,color='#3d5266',style='italic')
arr(ax,16.2,7.9,16.7,7.9); arr(ax,16.2,4.1,16.7,4.1)
arr(ax,6.35,2.95,16.7,2.95); arr(ax,6.35,0.95,16.7,0.95)
arr(ax,19.25,7.35,19.65,7.35); arr(ax,20.97,6.2,20.97,5.2)
ax.annotate('',xy=(18.0,0.3),xytext=(20.97,2.6),arrowprops=dict(arrowstyle='-|>',color='#7a8794',lw=0.8,ls='--',mutation_scale=7,connectionstyle='angle,angleA=-90,angleB=0,rad=0'))
ax.text(19.35,1.35,'delayed\nfeedback',fontsize=8,color='#5a6875',va='center')
save(fig,'Fig1')

# ---------------- Supplementary Figure S1: causal VMD, India, window ending 2021-12-30
z=np.log1p(ma['India']); end=pd.Timestamp('2021-12-30'); w=z.loc[:end].iloc[-128:]
f=w.values-w.values.mean()
u,_,omega=VMD(f,2000,0.0,4,0,1,1e-7)
order=np.argsort(omega[-1]); u=u[order]
n=min(len(w),u.shape[1]); idx=w.index[-n:]
fig,axs=plt.subplots(5,1,figsize=(COL,COL*1.45),sharex=True)
axs[0].plot(w.index,w.values,color='black',lw=0.8); axs[0].set_ylabel('log(1 + y)')
cols=['#08306b','#2171b5','#4292c6','#6baed6']
for k in range(4):
    axs[k+1].plot(idx,u[k][-n:],color=cols[k],lw=0.8); axs[k+1].set_ylabel(f'Mode {k+1}'); axs[k+1].axhline(0,color='#aaa',lw=0.4)
for a in axs: a.grid(color='#eee',lw=0.4); a.yaxis.set_label_coords(-0.2,0.5)
axs[-1].xaxis.set_major_locator(mdates.MonthLocator()); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
fig.align_ylabels(axs); fig.tight_layout(h_pad=0.4); save(fig,'FigS1')

# ---------------- Figure 2: DM heatmap
S=['−2.06**','−2.29**','−3.10***','−0.60','−2.71***','−5.06***','−4.06***','−10.49***',
'−0.94','−2.35**','0.07','−6.88***','−2.52**','−5.09***','−2.03**','−14.80***',
'−0.49','−2.74***','−2.11**','−5.04***','−2.52**','−5.39***','−1.84*','−0.69',
'−1.27','−4.88***','1.71*','−6.62***','0.67','−4.89***','−2.85***','−15.16***',
'−0.91','−1.50','−0.62','−2.68***','−37.88***','−4.11***','−6.45***','−7.75***',
'−1.69*','−1.61','−1.13','−4.40***','−43.54***','−5.05***','−8.98***','−27.73***',
'−0.82','−2.31**','0.10','−6.72***','−11.70***','−7.63***','0.83','−12.12***',
'0.98','−2.89***','−1.30','0.22','−17.40***','1.03','−0.51','−4.06***']  # DM statistics (Table S6)
V=np.array([float(s.rstrip('*').replace('−','-')) for s in S]).reshape(8,8); L=np.array(S).reshape(8,8)
series=['India\ndaily','Brazil\ndaily','Russia\ndaily','USA\ndaily','India\ncum.','Brazil\ncum.','Russia\ncum.','USA\ncum.']
rows=['Naive','ARIMA','SARIMA','ETS (damped)','LSTM','GRU','ARIMA–LSTM','DHBO-v1']
fig,ax=plt.subplots(figsize=(FULL,FULL*0.43))
im=ax.imshow(np.clip(V,-6,6),cmap='RdBu_r',vmin=-6,vmax=6,aspect='auto')
for i in range(8):
    for j in range(8): ax.text(j,i,L[i,j],ha='center',va='center',fontsize=8,color='white' if abs(V[i,j])>4 else '#1b1b1b')
ax.set_xticks(range(8)); ax.set_xticklabels(series); ax.set_yticks(range(8)); ax.set_yticklabels(['vs '+r for r in rows])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.axhline(6.5,color='white',lw=2)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.015); cb.set_label('DM statistic (clipped at ±6)'); cb.outline.set_visible(False)
fig.tight_layout(); save(fig,'Fig2')

# ---------------- Figure 3: coverage
ser=['India daily','Brazil daily','Russia daily','USA daily','India cum.','Brazil cum.','Russia cum.','USA cum.']
sp80=[.674,.686,.758,.711,.944,.901,.515,.972]; ac80=[.772,.767,.756,.762,.952,.844,.808,.866]
sp95=[.885,.929,.929,.925,.975,.990,.660,.998]; ac95=[.940,.941,.932,.934,.978,.964,.937,.976]
fig,axs=plt.subplots(1,2,figsize=(FULL,FULL*0.36),sharey=True); y=np.arange(8)[::-1]
for ax,sp,ac,nom,t in [(axs[0],sp80,ac80,.80,'Nominal 80%'),(axs[1],sp95,ac95,.95,'Nominal 95%')]:
    ax.axvline(nom*100,color='#555',lw=0.7,ls='--')
    for yi,a,b in zip(y,sp,ac): ax.plot([a*100,b*100],[yi,yi],color='#c9c9c9',lw=0.9,zorder=1)
    ax.scatter(np.array(sp)*100,y,s=18,facecolor='white',edgecolor='#E69F00',lw=1.1,label='Split conformal',zorder=3)
    ax.scatter(np.array(ac)*100,y,s=18,marker='D',color='#0072B2',label='Adaptive conformal (ACI)',zorder=3)
    ax.set_title(t); ax.set_xlabel('Empirical coverage (%)'); ax.grid(axis='x',color='#eee',lw=0.4)
axs[0].set_yticks(y); axs[0].set_yticklabels(ser); axs[0].set_xlim(48,100); axs[1].set_xlim(63,101)
axs[1].legend(loc='lower left',frameon=False); fig.tight_layout(); save(fig,'Fig3')

# ---------------- Figure 4: relative MASE
order=['DHBO-v2','DHBO-v2-CNN','DHBO-v1','ARIMA–LSTM hybrid','SARIMA','ARIMA','ETS (damped)','GRU','LSTM','Naive']
d=dict(dev); d['DHBO-v2']=0.784; d['ARIMA–LSTM hybrid']=1.130  # unrounded values (Table 7); the rounded tables give 0.785 and 1.129
fig,ax=plt.subplots(figsize=(FULL,FULL*0.39)); y=np.arange(len(order))[::-1]
for yi,m in zip(y,order): ax.plot([d[m],held[m]],[yi,yi],color='#cfcfcf',lw=0.9,zorder=1)
ax.scatter([d[m] for m in order],y,s=20,color='#0072B2',label='Development countries (IN, BR, RU, US)',zorder=3)
ax.scatter([held[m] for m in order],y,s=20,marker='s',facecolor='white',edgecolor='#D55E00',lw=1.1,label='Held-out countries (FR, DE, JP, KR)',zorder=3)
ax.axvline(1,color='#555',lw=0.7,ls='--'); ax.text(1.01,len(order)-0.4,'SARIMA = 1',color='#555')
ax.set_yticks(y); ax.set_yticklabels(['DHBO-v2 (TCN, proposed)' if m=='DHBO-v2' else m for m in order]); ax.get_yticklabels()[0].set_weight('bold')
ax.set_xscale('log'); ax.set_xticks([0.7,0.8,1,1.25,1.5,2]); ax.set_xticklabels(['0.7','0.8','1.0','1.25','1.5','2.0']); ax.minorticks_off()
ax.set_xlabel('Relative MASE (geometric mean over 8 series; lower is better)'); ax.grid(axis='x',color='#eee',lw=0.4)
ax.legend(loc='upper right',frameon=True,framealpha=1,edgecolor='#ddd'); fig.tight_layout(); save(fig,'Fig4')

# ---------------- Figure 5: reporting regimes
fig,axs=plt.subplots(2,2,figsize=(FULL,FULL*0.54))
for r,(c,lab) in enumerate([('Brazil','Brazil'),('US','USA')]):
    for k,(a,b,tt) in enumerate([('2021-06-01','2021-08-31','training sample (Jun–Aug 2021)'),('2022-10-01','2022-12-31','test sample (Oct–Dec 2022)')]):
        ax=axs[r,k]; dd=daily.loc[a:b,c]/1e3; mm=ma.loc[a:b,c]/1e3
        ax.bar(dd.index,dd.values,width=0.8,color='#bdbdbd',label='Reported daily cases')
        ax.plot(mm.index,mm.values,color='#1f5aa6',lw=1.0,label='7-day mean (model input)')
        ax.set_title(f'{lab}: {tt}'); ax.grid(axis='y',color='#eee',lw=0.4)
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        if k==0: ax.set_ylabel('Cases (thousands)')
axs[0,0].legend(loc='upper right',frameon=False); fig.tight_layout(); save(fig,'Fig5')
print('ok')
