"""Reproduce the robustness figures from audited, frozen inputs; no network access."""
from pathlib import Path
import argparse,csv,json,shutil,hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from robustness_style import *
HERE=Path(__file__).resolve().parent
PORTABLE=HERE.name=='source'
DESIGN=HERE.parent
PACKAGE=DESIGN if PORTABLE else DESIGN/'overleaf-section5'
FIG=PACKAGE/'figures' if PORTABLE else HERE
PDF=FIG if PORTABLE else DESIGN.parents[1]/'output/pdf/section5-robustness'
PDF.mkdir(parents=True,exist_ok=True);(PACKAGE/'figures').mkdir(parents=True,exist_ok=True)
D=json.loads((DESIGN/'snapshot/govsim/figure-data.json').read_text())
T=json.loads((DESIGN/'snapshot/evals-out-p2/analysis-io/tables.json').read_text())
ORDER=['A','Aio','MIS']
MIX={'A':'BBBBA-tf','Aio':'BBBBa-tf','MIS':'BBBBm-tf'}
UNIFORM={'A':'AAAAA-tf','Aio':'aaaaa-tf','MIS':'mmmmm-tf'}
MONTHS=np.arange(1,13)
PLOTDATA={'resource':{},'behavior':{},'belief_trust':{}}
CATEGORIES=[
 ('plan_mislead_month1','Month-1 private plan to mislead'),
 ('plan_mislead','Any decision planning to mislead'),
 ('cooperative_framing','Private text calls its own catch cooperative'),
 ('first_month_request_over_share','Month-1 catch above the 10 t share'),
 ('commitment_breach',"Next catch above its own promise"),
 ('agreement_breach',"Next catch above the group's agreed limit"),
 ('explanation_after_mislead_or_exploit_plan','Excuses that follow a private plan to mislead or exploit'),
 ('excusing','Excusing turns'),
 ('apology','Apologies'),
 ('false_self_claim','Turns with a false claim about himself'),
 ('survived','Group survives 12 months')]

def summarize(vals):
 vals=np.asarray(vals,dtype=float)
 return vals.mean(axis=0),vals.std(axis=0,ddof=1)/np.sqrt(len(vals))

def stock(ax,comps,title,baseline=False):
 mapping=({'base':'BBBBB-tf',**comps} if baseline else comps)
 for arm,comp in mapping.items():
  rows=sorted([r for r in D['runs'] if r['comp']==comp],key=lambda r:(r['training_seed'],r['simulation_seed']))
  assert len(rows)==(5 if arm=='base' else 15)
  mat=np.zeros((len(rows),12));seeds=np.array([r['training_seed'] for r in rows])
  for i,r in enumerate(rows):
   ts=r['resource_trajectory'];assert [x['month'] for x in ts]==list(range(1,len(ts)+1))
   if len(ts)<12:assert ts[-1]['stock_after']<=5
   for m in ts: mat[i,m['month']-1]=m['stock_before']
  seedmeans=np.array([mat[seeds==s].mean(0) for s in sorted(set(seeds))])
  mean=seedmeans.mean(0);sem=None if arm=='base' else summarize(seedmeans)[1]
  PLOTDATA['resource'][comp]={'mean':mean.tolist(),'seed_means':seedmeans.tolist(),'sem':None if sem is None else sem.tolist(),
    'survive_12':sum(r['metrics']['survived'] for r in rows),'n':len(rows)}
  if sem is not None:ax.fill_between(MONTHS,np.maximum(mean-sem,0),np.minimum(mean+sem,100),color=FILL[arm],alpha=.68,lw=0,zorder=2)
  ax.plot(MONTHS,mean,color=LINE[arm],ls=LS[arm],lw=1.3,marker=MARKER[arm],ms=2.1,mew=0,zorder=4)
 ax.set(xlim=(.8,12.2),ylim=(-2,103),xticks=[1,3,6,9,12],yticks=[0,25,50,75,100],xlabel='Month',ylabel='Fish stock (tons)')
 ax.grid(axis='y');ax.set_title(title,loc='left',fontweight='bold',pad=8)

def behavior_values():
 # Each category retains the source table's numerator and eligible denominator.
 # These are descriptive pooled proportions: no sampling intervals are plotted.
 for arm in ORDER:
  comp=MIX[arm]
  rows=sorted([r for r in D['runs'] if r['comp']==comp],key=lambda r:(r['training_seed'],r['simulation_seed']))
  assert len(rows)==15
  vals=np.zeros((3,5,len(CATEGORIES),2))
  for i,r in enumerate(rows):
   for j,(key,_) in enumerate(CATEGORIES):
    vals[i//5,i%5,j]=[r['metrics']['survived'],1] if key=='survived' else [r['behavior'][key]['count'],r['behavior'][key]['denominator']]
  seed_counts=vals.sum(1)
  seed_rates=seed_counts[:,:,0]/seed_counts[:,:,1]*100
  total=vals.sum((0,1));point=total[:,0]/total[:,1]*100
  for i,(key,label) in enumerate(CATEGORIES):
   if key!='survived':
    summary=D['summaries'][comp][key]
    assert int(total[i,0])==summary['pooled_count'] and int(total[i,1])==summary['pooled_denominator']
   PLOTDATA['behavior'][f'{comp}:{key}:table_rate']={'label':label,'count':int(total[i,0]),'denominator':int(total[i,1]),
      'percent':float(point[i]),'seed_percentages':seed_rates[:,i].tolist()}

def behavior(ax,categories,title):
 import textwrap
 # Each category has three distinct bars; printed values also make zeroes visible.
 positions=list(range(len(categories)))
 if categories[-1][0]=='survived':positions=[0,1.25,2.25,3.25,4.65]
 for row,(key,label) in zip(positions,categories):
  lines=textwrap.fill(label,49,break_long_words=False,break_on_hyphens=False)
  if key=='explanation_after_mislead_or_exploit_plan':lines='Excuses that follow a private plan\nto mislead or exploit'
  ax.text(0,row,lines,fontsize=6.5,ha='left',va='top',linespacing=1.1)
  offset=.59 if '\n' in lines else .35
  spacing=.21
  if key=='survived':
   offset=.43;spacing=.31
   ax.axhline(row-.17,color='#AAAAAA',lw=.5)
  bar_y=row+offset
  for x in [0,50,100]:
   ax.plot([x,x],[bar_y-.095,bar_y+2*spacing+.095],color='#CBCBCB',ls=':',lw=.6,zorder=1)
  for ai,arm in enumerate(ORDER):
   value=PLOTDATA['behavior'][f'{MIX[arm]}:{key}:table_rate']
   point=value['percent'];y=bar_y+ai*spacing
   if point>0:ax.barh(y,point,height=.17,color=FILL[arm],edgecolor=INK,lw=.45,zorder=3)
   display=f'{point:.0f}%'
   if key=='survived':display=f"{value['count']}/{value['denominator']} ({point:.0f}%)"
   ax.text(point+2.4,y,display,fontsize=5.8 if key!='survived' else 6.1,
           ha='left',va='center',color=INK)
 ax.set(yticks=[],xlim=(0,121),xticks=[0,50,100],ylim=(5.98,-.10))
 ax.set_xlabel('Proportion (%)',labelpad=3)
 for side in ['top','right','left']:ax.spines[side].set_visible(False)
 ax.set_title(title,loc='left',fontweight='bold',pad=8)

def export_img(fig,stem):
 for ext in ['png','svg']: fig.savefig(FIG/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.03)
 if not PORTABLE:
  for ext in ['png','svg']:shutil.copyfile(FIG/f'{stem}.{ext}',PACKAGE/f'figures/{stem}.{ext}')

def govsim():
 dest=PDF/'section5-govsim.pdf'
 with PdfPages(dest) as pdf:
  fig=plt.figure(figsize=(5.5,3.85))
  left=fig.add_axes([.035,.19,.445,.70])
  right=fig.add_axes([.54,.19,.445,.70])
  behavior_values()
  behavior(left,CATEGORIES[:6],'A  Private plans and extraction')
  behavior(right,CATEGORIES[6:],'B  Public responses and survival')
  legend(fig,y=.026)
  export_img(fig,'section5-govsim-main');pdf.savefig(fig,bbox_inches='tight',pad_inches=.03);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(5.5,2.65),sharey=True,
      gridspec_kw={'left':.095,'right':.99,'top':.86,'bottom':.28,'wspace':.15})
  stock(axes[0],MIX,'A  One replacement')
  stock(axes[1],UNIFORM,'B  Homogeneous groups')
  axes[1].set_ylabel('')
  legend(fig,y=.025)
  export_img(fig,'section5-govsim-appendix');pdf.savefig(fig,bbox_inches='tight',pad_inches=.03);plt.close(fig)
 if dest.resolve()!=(PACKAGE/'figures'/dest.name).resolve():shutil.copyfile(dest,PACKAGE/'figures'/dest.name)

def cell_rows(arm):
 return [T['A']['base']] if arm=='base' else T['A']['main'] if arm=='A' else T['A']['io'] if arm=='Aio' else T['sfm']['MIS']

def metric_bars(ax,keys,labels,title,ylabel,ylim,yticks,scales=None):
 scales=scales or [1]*len(keys)
 for ai,arm in enumerate(['base',*ORDER]):
  for j,(key,scale) in enumerate(zip(keys,scales)):
   values=np.array([r[key]*scale for r in cell_rows(arm)],dtype=float)
   assert np.isfinite(values).all()
   x=j+(ai-1.5)*.20;mean=values.mean()
   ax.bar(x,mean,width=.18,color=FILL[arm],edgecolor=INK,lw=.55,zorder=3)
   if arm!='base':
    sem=summarize(values)[1]
    ax.errorbar(x,mean,yerr=sem,fmt='none',ecolor=INK,elinewidth=.65,capsize=1.6,capthick=.65,zorder=4)
    ax.scatter(x+np.linspace(-.03,.03,3),values,s=6,c=INK,lw=0,zorder=5)
   PLOTDATA['belief_trust'][f'{arm}:{key}']={'seed_values':values.tolist(),'mean':float(mean),'sem':None if arm=='base' else float(sem)}
 ax.set(xticks=range(len(keys)),xticklabels=labels,xlim=(-.55,len(keys)-.45),ylim=ylim,yticks=yticks,ylabel=ylabel)
 ax.tick_params(axis='x',length=0,pad=5,labelsize=6.1);ax.grid(axis='y');ax.set_title(title,loc='left',fontweight='bold',pad=8)

def belief_trust():
 fig,ax=plt.subplots(2,2,figsize=(5.5,4.9),gridspec_kw={'left':.09,'right':.99,'top':.925,'bottom':.15,'hspace':.55,'wspace':.4})
 metric_bars(ax[0,0],['endorse_A','alloc_A'],['Antisocial claim\nendorsement','Antisocial\nallocation'],'A  Antisocial population beliefs','Percent',(0,105),[0,25,50,75,100],[100,1])
 metric_bars(ax[0,1],['yam_people','yam_ai_users'],['People\nin general','AI-assistant\nusers'],'B  Generalized trust','Trust (1–5)',(0,5.2),[0,1,2,3,4,5])
 fr=['person','ai_user','ai_agent'];labels=['Person','AI-assistant\nuser','AI agent']
 metric_bars(ax[1,0],['v2_none_'+f+'_ev' for f in fr],labels,'C  Expected return, no record','Expected return ($ of 30)',(0,17),[0,5,10,15])
 metric_bars(ax[1,1],['v2_none_'+f+'_act' for f in fr],labels,'D  Amount sent, no record','Amount sent ($ of 10)',(0,10.5),[0,2,4,6,8,10])
 legend(fig,keys=['base',*ORDER],y=.025)
 dest=PDF/'section5-belief-trust.pdf';fig.savefig(dest,bbox_inches='tight',pad_inches=.03)
 export_img(fig,'section5-belief-trust');plt.close(fig)
 if dest.resolve()!=(PACKAGE/'figures'/dest.name).resolve():shutil.copyfile(dest,PACKAGE/'figures'/dest.name)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--only',choices=['govsim','belief-trust'],help='Regenerate only one figure; retain other existing plotting values.')
 args=parser.parse_args()
 numbers=HERE/'robustness-plot-numbers.json'
 if args.only and numbers.exists():PLOTDATA.update(json.loads(numbers.read_text()))
 if args.only!='belief-trust':
  PLOTDATA['behavior']={}
  govsim()
 if args.only!='govsim':belief_trust()
 numbers.write_text(json.dumps(PLOTDATA,indent=2)+'\n')
 print('Created requested robustness figures; frozen data verified.')
