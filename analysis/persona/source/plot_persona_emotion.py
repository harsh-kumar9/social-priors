"""Paper-theme persona/emotion figures from audited checkpoint-level readouts."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator,FixedFormatter
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'figures/persona-emotion-data.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'figures');args=p.parse_args();OUT=args.output_dir;OUT.mkdir(exist_ok=True,parents=True)
FILL=dict(P='#C7D9E9',S='#F0DFAE',N='#D6D6D2',A='#E7C2BB',base='#FFFFFF',MIX='#DDDDDD',REPLAY='#F4F4F4')
LINE=dict(P='#37668F',S='#876A16',N='#5F5F5B',A='#A14C42',base='#242424',MIX='#747474',REPLAY='#999999')
INK='#242424';EMO=['anger','disgust','fear','sadness','joy','surprise'];ALL=['base','P','S','N','A','MIX','REPLAY'];WORLD=['P','S','N','A']
NAME=dict(base='Qwen3-32B',P='Prosocial',S='Self-advancing',N='Task-instrumental',A='Antisocial',MIX='Mixed worlds',REPLAY='Web text')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.2,'axes.labelsize':6.6,'xtick.labelsize':6,'ytick.labelsize':6.2,'legend.fontsize':6,'axes.edgecolor':INK,'axes.linewidth':.7,'axes.axisbelow':True,'grid.color':'#CBCBCB','grid.linestyle':':','grid.linewidth':.6,'xtick.major.size':2.5,'ytick.major.size':0,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'legend.frameon':False,'legend.handlelength':1.3,'legend.handletextpad':.45,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':360,'figure.facecolor':'white','axes.facecolor':'white'})

def meansem(v):
 v=np.asarray(v,float);return v.mean(),v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0

def vals(mode,arm,key):return [r['metrics'][key] for r in D[mode] if r['arm']==arm and key in r['metrics']]
def bar(ax,x,v,arm,width=.20):
 m,se=meansem(v);ax.bar(x,m,width=width,color=FILL[arm],edgecolor=INK,lw=.55,zorder=3)
 if len(v)>1:
  ax.errorbar(x,m,yerr=se,fmt='none',ecolor=INK,elinewidth=.6,capsize=1.5,capthick=.6,zorder=4)
  ax.scatter(x+np.linspace(-.025,.025,len(v)),v,s=5,color=INK,lw=0,zorder=5)

def title(ax,t):ax.set_title(t,loc='left',fontweight='bold',pad=7)
def legend(fig,arms,y=.02,ncol=None):
 fig.legend(handles=[Patch(facecolor=FILL[a],edgecolor=INK,lw=.6,label=NAME[a]) for a in arms],loc='lower center',bbox_to_anchor=(.52,y),ncol=ncol or len(arms),columnspacing=1.0)
def save(fig,pdf,stem):
 pdf.savefig(fig,bbox_inches='tight',pad_inches=.035)
 for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.035)
 plt.close(fig)
def capbars(ax,key,arms,ttl,ylabel,lim):
 for i,mode in enumerate(['natural','cap']):
  for j,a in enumerate(arms):bar(ax,i+(j-(len(arms)-1)/2)*.22,vals(mode,a,key),a)
 ax.set(xticks=[0,1],xticklabels=['No cap','Capped'],ylabel=ylabel,ylim=lim,xlim=(-.5,1.5));ax.grid(axis='y');title(ax,ttl)

def drift(ax,arms,ttl):
 b=vals('natural','base','axis_response')[0]
 for i,a in enumerate(arms):bar(ax,i,[b-x for x in vals('natural',a,'axis_response')],a,.64)
 ax.set(xticks=range(len(arms)),xticklabels=['B' if a=='base' else 'I' if a=='N' else 'Mix' if a=='MIX' else 'Web' if a=='REPLAY' else a for a in arms],ylabel='Axis displacement\nfrom reference',ylim=(-.18,5.3));ax.grid(axis='y');title(ax,ttl)

def curves(direction='valence@L22-24',xmode='measured'):
 runs=[r for r in D['steering'] if r['direction']==direction and r['arm'] in ['P','A']]
 alphas=sorted(set(r['alpha'] for r in runs));zero={a:np.mean([r['metrics']['emo_proj_action'] for r in runs if r['arm']==a and r['alpha']==0]) for a in ['P','A']}
 gap=zero['A']-zero['P'];x=[]
 for alpha in alphas:
  if xmode=='measured':
   x.append(np.mean([(np.mean([r['metrics']['emo_proj_action'] for r in runs if r['arm']==a and r['alpha']==alpha])-zero[a])/gap for a in ['P','A']]))
  else:
   f=next(r['frozen'] for r in runs if r['alpha']==alpha);x.append(alpha/f['alpha_method_scale'])
 return runs,alphas,np.array(x)

def linepanel(ax,key,ttl,ylabel,lim,direction='valence@L22-24',xmode='measured',show_dots=False):
 runs,alphas,x=curves(direction,xmode)
 if xmode=='measured':ax.axvspan(-4,4,color='#F2F2F2',lw=0,zorder=0)
 for a in ['P','A']:
  vs=[[r['mean_letter_mass'] if key=='letter_mass' else r['metrics'][key] for r in runs if r['arm']==a and r['alpha']==alpha] for alpha in alphas]
  factor=100 if key in ['own_world','letter_mass'] else 1
  means=np.array([meansem(v)[0] for v in vs])*factor;ses=np.array([meansem(v)[1] for v in vs])*factor
  ax.fill_between(x,means-ses,means+ses,color=FILL[a],alpha=.7,lw=0,zorder=2)
  ax.plot(x,means,'-o',color=LINE[a],lw=1.2,ms=2.3,zorder=4)
  if show_dots:
   for xx,yy in zip(x,vs):ax.scatter([xx]*len(yy),np.array(yy)*factor,color=INK,s=3,lw=0,zorder=3)
 if xmode=='measured':
  ax.set_xscale('symlog',linthresh=4,linscale=.8);ax.set_xlim(-95,95)
  ax.xaxis.set_major_locator(FixedLocator([-80,-4,0,4,80]));ax.xaxis.set_major_formatter(FixedFormatter(['-80','-4','0','4','80']));ax.minorticks_off()
  ax.set_xlabel('Measured affect shift\n(natural gap units)',labelpad=3)
 else:ax.set(xlim=(-2.15,2.15),xticks=[-2,0,2],xlabel='Method-scale coefficient')
 ax.set(ylabel=ylabel,ylim=lim);ax.grid(axis='y');title(ax,ttl)

# Main figure: the two tests, each with its observational context and interventions.
with PdfPages(OUT/'persona-emotion-main.pdf') as pdf:
 fig,ax=plt.subplots(2,3,figsize=(5.5,4.65),gridspec_kw={'left':.08,'right':.99,'top':.94,'bottom':.19,'hspace':.65,'wspace':.55})
 drift(ax[0,0],['base',*WORLD],'A  Assistant-axis drift')
 # Belief probabilities are displayed as percentages.
 for i,mode in enumerate(['natural','cap']):
  for j,a in enumerate(['P','A']):bar(ax[0,1],i+(j-.5)*.25,np.array(vals(mode,a,'own_world'))*100,a,.23)
 ax[0,1].set(xticks=[0,1],xticklabels=['No cap','Capped'],ylim=(0,105),xlim=(-.5,1.5),ylabel='World endorsement (%)',yticks=[0,50,100]);ax[0,1].grid(axis='y');title(ax[0,1],'B  Belief under capping')
 capbars(ax[0,2],'amount_ev',['base','P','A'],'C  Transfer under capping','Scored amount ($)',(0,7))
 for i,e in enumerate(EMO):
  r=D['emotion_gaps'][e];y=5-i
  ax[1,0].barh(y,r['half2'],height=.67,color='#D6D6D2',edgecolor=INK,lw=.55,zorder=3)
  ax[1,0].scatter(r['half1'],y,s=10,facecolor='white',edgecolor=INK,lw=.6,zorder=4)
 ax[1,0].axvline(0,color=INK,lw=.6);ax[1,0].set(yticks=range(6),yticklabels=list(reversed(EMO)),xlim=(-5.6,5.6),xticks=[-5,0,5],xlabel='A-P projection gap\n(pooled item SDs)',ylim=(-.5,5.5));ax[1,0].grid(axis='x');title(ax[1,0],'D  Emotion projections')
 linepanel(ax[1,1],'amount_ev','E  Transfer under steering','Scored amount ($)',(0,7))
 linepanel(ax[1,2],'own_world','F  Belief under steering','World endorsement (%)',(0,105))
 ax[1,2].set_yticks([0,50,100]);legend(fig,['base',*WORLD],y=.015)
 save(fig,pdf,'persona-emotion-main')

with PdfPages(OUT/'persona-emotion-appendix.pdf') as pdf:
 # 1. All-condition axis readout and cap engagement.
 fig,ax=plt.subplots(2,2,figsize=(5.5,4.6),gridspec_kw={'left':.09,'right':.99,'top':.93,'bottom':.13,'hspace':.58,'wspace':.35})
 drift(ax[0,0],ALL,'A  Assistant-axis drift')
 for i,a in enumerate(ALL):
  ds=[r for r in D['cap'] if r['arm']==a]
  for j,phase in enumerate(['last_prompt_token','generated_tokens']):
   v=[100*r['cap_stats'][phase]['frac_positions_capped'] for r in ds]
   bar(ax[0,1],i+(j-.5)*.34,v,a,.3)
   if j==0:
    for patch in ax[0,1].patches[-1:]:patch.set_hatch('///')
 ax[0,1].set(xticks=range(7),xticklabels=['B','P','S','I','A','Mix','Web'],ylim=(0,103),ylabel='Capped token-layer positions (%)');ax[0,1].grid(axis='y');title(ax[0,1],'B  Cap engagement')
 ax[0,1].legend(handles=[Patch(facecolor='white',edgecolor=INK,hatch='///',label='Last prompt token'),Patch(facecolor='white',edgecolor=INK,label='Generated tokens')],loc='upper left',fontsize=5.5,handlelength=1)
 for col,key in enumerate(['own_world','amount_ev']):
  arms=WORLD if key=='own_world' else ALL
  for i,a in enumerate(arms):
   for j,mode in enumerate(['natural','cap']):
    v=np.array(vals(mode,a,key))*(100 if key=='own_world' else 1);bar(ax[1,col],i+(j-.5)*.33,v,a,.29)
    if j==1:ax[1,col].patches[-1].set_hatch('///')
  ax[1,col].set(xticks=range(len(arms)),xticklabels=['B' if a=='base' else 'I' if a=='N' else 'Mix' if a=='MIX' else 'Web' if a=='REPLAY' else a for a in arms],ylim=(0,105) if key=='own_world' else (0,7),ylabel='World endorsement (%)' if key=='own_world' else 'Scored amount ($)');ax[1,col].grid(axis='y')
  title(ax[1,col],'C  Belief under capping' if col==0 else 'D  Transfer under capping')
 fig.legend(handles=[Patch(facecolor='white',edgecolor=INK,label='No cap (C,D)'),Patch(facecolor='white',edgecolor=INK,hatch='///',label='Capped (C,D)')],loc='lower center',bbox_to_anchor=(.54,.062),ncol=2,fontsize=5.8)
 fig.text(.53,.025,'B: Qwen3-32B; I: task-instrumental; Mix: mixed worlds; Web: web text',ha='center',fontsize=6)
 save(fig,pdf,'persona-controls-appendix')
 # 2. All four world profiles, at selected peaks and common extraction layers.
 fig,axes=plt.subplots(2,1,figsize=(5.5,4.5),gridspec_kw={'left':.12,'right':.99,'top':.92,'bottom':.16,'hspace':.52})
 for ax,window,ttl in zip(axes,['peak','method'],['A  Emotion-specific peak layers','B  Common layers 22-24']):
  for i,e in enumerate(EMO):
   row=D['emotion_profiles'][window+':'+e]
   for j,a in enumerate(WORLD):bar(ax,i+(j-1.5)*.19,row['arms'][a]['z_seed_means'],a,.17)
  ax.axhline(0,color=INK,lw=.65)
  labels=[e+('\nL'+str(D['emotion_gaps'][e]['layer']) if window=='peak' else '') for e in EMO]
  ax.set(xticks=range(6),xticklabels=labels,ylabel='Projection relative to reference\n(pooled P/A item SDs)');ax.grid(axis='y');title(ax,ttl)
 legend(fig,WORLD,y=.015);save(fig,pdf,'emotion-profiles-appendix')
 # 3. Manipulation checks and the full composite sweep.
 fig,ax=plt.subplots(2,2,figsize=(5.5,4.9),gridspec_kw={'left':.1,'right':.99,'top':.93,'bottom':.15,'hspace':.63,'wspace':.38})
 runs,alphas,x=curves()
 for a in ['P','A']:
  zero=np.mean([r['metrics']['emo_proj_action'] for r in runs if r['arm']==a and r['alpha']==0])
  gap=np.mean([r['metrics']['emo_proj_action'] for r in runs if r['arm']=='A' and r['alpha']==0])-np.mean([r['metrics']['emo_proj_action'] for r in runs if r['arm']=='P' and r['alpha']==0])
  yy=[(np.mean([r['metrics']['emo_proj_action'] for r in runs if r['arm']==a and r['alpha']==al])-zero)/gap for al in alphas]
  ax[0,0].plot(np.array(alphas)/D['frozen']['alpha_unit_raw'],yy,'-o',color=LINE[a],lw=1.2,ms=2.6)
 ax[0,0].plot([-42,42],[-42,42],':',color='#777777',lw=.7)
 ax[0,0].set(xlabel='Nominal dose (gap units per layer)',ylabel='Measured shift (gap units)');ax[0,0].grid();title(ax[0,0],'A  Manipulation check')
 linepanel(ax[0,1],'amount_ev','B  Transfer score','Scored amount ($)',(0,7),show_dots=True)
 linepanel(ax[1,0],'own_world','C  World endorsement','World endorsement (%)',(0,105),show_dots=True)
 linepanel(ax[1,1],'letter_mass','D  Answer-letter probability mass','Probability mass (%)',(95,100.3),show_dots=True)
 legend(fig,['P','A'],y=.01);save(fig,pdf,'emotion-composite-appendix')
 # 4. Individual directions at the common extraction layers.
 fig,ax=plt.subplots(6,2,figsize=(5.5,7.8),gridspec_kw={'left':.1,'right':.99,'top':.97,'bottom':.11,'hspace':.72,'wspace':.32})
 for i,e in enumerate(EMO):
  for j,key in enumerate(['amount_ev','own_world']):
   linepanel(ax[i,j],key,e.capitalize() if j==0 else '', 'Scored $' if j==0 else 'Belief (%)',(0,10) if j==0 else (0,105),direction=e+'@L22-24',xmode='method')
   ax[i,j].set_xlabel('' if i<5 else 'Method-scale coefficient');ax[i,j].set_yticks([0,5,10] if j==0 else [0,50,100])
   if i<5:ax[i,j].tick_params(axis='x',labelbottom=False)
 legend(fig,['P','A'],y=.01);save(fig,pdf,'emotion-individual-appendix')
 # 5. Where/when to intervene; each column has its own method-scale normalization.
 fig,ax=plt.subplots(2,3,figsize=(5.5,4.1),gridspec_kw={'left':.085,'right':.99,'top':.90,'bottom':.17,'hspace':.42,'wspace':.43})
 for j,(direction,ttl) in enumerate([('valence@L22-24|gen','Last position, L22-24'),('valence@L32-34','All positions, L32-34'),('valence@L46-48','All positions, L46-48')]):
  for i,key in enumerate(['amount_ev','own_world']):
   linepanel(ax[i,j],key,ttl if i==0 else '', 'Scored $' if i==0 else 'Belief (%)',(0,10) if i==0 else (0,105),direction=direction,xmode='method')
   if i==0:ax[i,j].set_xlabel('')
   if j>0:ax[i,j].set_ylabel('')
 legend(fig,['P','A'],y=.01);save(fig,pdf,'emotion-layers-appendix')
print('Created main figure and five-page appendix PDF.')
