"""Restore all three original Section 3 figure families in Section 2's style.

Input is the live Ada snapshot ../source-data.json. Outputs PNG/SVG previews in
this directory and final PDFs under output/pdf/section3-broad. Earlier figures
and the independent forecast/action design prototypes are never overwritten.
"""
from pathlib import Path
import json, csv, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE=Path(__file__).resolve().parent
SOURCE_PATH=HERE.parent/'source-data.json'
SOURCE=json.loads(SOURCE_PATH.read_text())
WORKSPACE=HERE.parent
PDF=WORKSPACE/'figures'
PDF.mkdir(parents=True,exist_ok=True)
KEYS={'trust':'evals-out-p2/trust_v2/trust_v2_tables.json','generalized':'evals-out-p2/analysis-32b/yamagishi.json','dictator':'evals-out-p2/trust/dictator_cells_32b.json'}
DATA={k:SOURCE['files'][p]['data'] for k,p in KEYS.items()}
PALETTE=HERE/'world_plot_style.py'
sys.path.insert(0,str(PALETTE.parent))
from world_plot_style import FILL, LINE, INK
ALL_ARMS=['base','P','S','N','A'];TRUST_ARMS=['base','P','A']
NAMES={'base':'Original model','P':'Prosocial (P)','S':'Legitimate self-\nadvancement (S)','N':'Task-instrumental (N)','A':'Antisocial (A)'}
NUMBERS={};VALIDATION=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,
 'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,'legend.fontsize':6.2,
 'axes.edgecolor':INK,'axes.linewidth':.7,'axes.axisbelow':True,
 'axes.spines.top':True,'axes.spines.right':True,'grid.color':'#CBCBCB',
 'grid.linestyle':':','grid.linewidth':.6,'xtick.major.size':2.5,'ytick.major.size':0,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'legend.frameon':False,'legend.handlelength':1.3,'legend.handletextpad':.45,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.6,
 'savefig.dpi':360,'figure.facecolor':'white','axes.facecolor':'white','text.parse_math':False})


def remember(source,condition,metric,arm,values,reference=None):
 vals=np.asarray(values,dtype=float)
 assert len(vals)==(1 if arm=='base' else 3),(source,condition,metric,arm,len(vals))
 mean=float(vals.mean());sem=float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None
 if reference is not None:
  assert np.isclose(mean,reference[0],rtol=1e-10),(source,condition,metric,arm,'mean')
  if sem is not None:assert np.isclose(sem,reference[1],rtol=1e-10),(source,condition,metric,arm,'sem')
 key='|'.join([source,condition,metric,arm])
 NUMBERS[key]={'source':source,'condition':condition,'metric':metric,'arm':arm,'mean':mean,'sem':sem,'seed_values':vals.tolist()}
 return vals,mean,sem


def trust(q,condition,metric,arm):
 v=DATA['trust'][q][f'{condition}|{metric}'][arm]
 return remember(f'trust_v2/{q}',condition,metric,arm,v[2],v)


def generalized(condition,arm):
 ids=['qwen3-32b-base'] if arm=='base' else [f'dw2-{arm}-32b-s{s}' for s in range(3)]
 vals=[DATA['generalized'][sid][condition] for sid in ids]
 assert all(DATA['generalized'][sid]['n']==12 for sid in ids)
 return remember('generalized_trust',condition,'rating_1_to_5',arm,vals)


def dictator(condition,arm):return remember('dictator_v1',condition,'amount_given',arm,DATA['dictator'][condition][arm])


def bar(ax,x,data,arm,width=.21,horizontal=False):
 vals,mean,sem=data
 if horizontal:
  ax.barh(x,mean,height=width,facecolor=FILL[arm],edgecolor=INK,lw=.6,zorder=2)
 else:ax.bar(x,mean,width=width,facecolor=FILL[arm],edgecolor=INK,lw=.6,zorder=2)
 if sem is not None:
  jitter=np.linspace(-width*.23,width*.23,len(vals))
  if horizontal:
   ax.errorbar(mean,x,xerr=sem,fmt='none',ecolor=INK,elinewidth=.65,capsize=1.7,capthick=.65,zorder=4)
   ax.scatter(vals,x+jitter,s=7,color=INK,linewidths=0,zorder=5)
  else:
   ax.errorbar(x,mean,yerr=sem,fmt='none',ecolor=INK,elinewidth=.65,capsize=1.7,capthick=.65,zorder=4)
   ax.scatter(x+jitter,vals,s=7,color=INK,linewidths=0,zorder=5)


def grouped(ax,cells,metric='act',horizontal=False):
 for i,(q,condition) in enumerate(cells):
  for offset,arm in zip([-.235,0,.235],TRUST_ARMS):
   data=trust(q,condition,metric,arm) if q!='dictator' else dictator(condition,arm)
   bar(ax,i+offset,data,arm,horizontal=horizontal)


def legend(fig,arms,anchor,ncol):
 fig.legend(handles=[Patch(facecolor=FILL[a],edgecolor=INK,lw=.6,label=NAMES[a]) for a in arms],
  loc='lower center',bbox_to_anchor=anchor,ncol=ncol,columnspacing=1.2)


def save(fig,stem):
 fig.savefig(PDF/f'{stem}.pdf',bbox_inches='tight',pad_inches=.03)
 for ext in ['png','svg']:fig.savefig(HERE/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.03)
 plt.close(fig)


def perceptions():
 fig,axes=plt.subplots(1,3,figsize=(5.5,2.75),
  gridspec_kw=dict(left=.078,right=.99,top=.86,bottom=.34,wspace=.34,width_ratios=[.96,.96,1.62]))
 for ax,condition,title in zip(axes[:2],['people','ai_users'],['A  People in general','B  AI-assistant users']):
  for i,arm in enumerate(ALL_ARMS):bar(ax,i,generalized(condition,arm),arm,width=.70)
  ax.set_xticks(range(5),['Orig.','P','S','N','A']);ax.set_xlim(-.6,4.6)
  ax.set_ylim(0,5.3);ax.set_yticks(range(6));ax.grid(axis='y')
  ax.set_title(title,loc='left',fontweight='bold',pad=8);ax.tick_params(axis='x',length=0,labelsize=6)
 axes[0].set_ylabel('Generalized trust (1-5 scale)',labelpad=3)
 axes[1].tick_params(labelleft=False)
 ax=axes[2]
 grouped(ax,[('q5',f'{label} none') for label in ['person','ai_user','ai_agent']],metric='pred_ev')
 ax.set_xticks(range(3),['Person','AI-assistant\nuser','AI agent']);ax.tick_params(axis='x',labelsize=6)
 ax.set_xlim(-.6,2.6);ax.set_ylim(0,30);ax.set_yticks([0,10,20,30]);ax.grid(axis='y')
 ax.set_ylabel('Expected return ($ of $30)',labelpad=3)
 ax.set_title('C  No counterpart record',loc='left',fontweight='bold',pad=8)
 legend(fig,ALL_ARMS,(.55,.025),5)
 save(fig,'section3-perceptions')


def decisions():
 fig,axes=plt.subplots(1,3,figsize=(5.5,2.95),sharey=True,
  gridspec_kw=dict(left=.07,right=.99,top=.80,bottom=.34,wspace=.14,width_ratios=[1.26,1.68,.98]))
 ax=axes[0]
 grouped(ax,[('q5',f'{label} none') for label in ['person','ai_user','ai_agent']])
 ax.set_xticks(range(3),['Person','AI-assistant\nuser','AI agent']);ax.set_xlim(-.6,2.6)
 ax.set_title('A  No counterpart record',loc='left',fontweight='bold',pad=25)
 ax=axes[1]
 grouped(ax,[('q1',c) for c in ['reciprocal three','reciprocal ten','exploitative three','exploitative ten']])
 ax.set_xticks(range(4),['3','10','3','10']);ax.set_xlim(-.6,3.6)
 ax.axvline(1.5,color='#ABABAB',lw=.6,ls=(0,(3,3)),zorder=1)
 ax.text(.5,11,'Reciprocal',ha='center',va='bottom',fontsize=6.1,clip_on=False)
 ax.text(2.5,11,'Exploitative',ha='center',va='bottom',fontsize=6.1,clip_on=False)
 ax.set_title('B  Person with a record',loc='left',fontweight='bold',pad=25)
 ax.set_xlabel('Prior exchanges shown',labelpad=4)
 ax=axes[2]
 grouped(ax,[('dictator','person'),('dictator','ai_user')])
 ax.set_xticks([0,1],['Person','AI-assistant\nuser']);ax.set_xlim(-.6,1.6)
 ax.set_title('C  Dictator game',loc='left',fontweight='bold',pad=25)
 for ax in axes:
  ax.set_ylim(-.1,10.7);ax.set_yticks([0,2,4,6,8,10]);ax.grid(axis='y');ax.tick_params(axis='x',labelsize=6)
 axes[0].set_ylabel('Amount sent or given ($ of $10)',labelpad=3)
 legend(fig,TRUST_ARMS,(.55,.025),3)
 save(fig,'section3-decisions')


def variations():
 fig,axes=plt.subplots(1,2,figsize=(5.5,3.35),sharey=True,
  gridspec_kw=dict(left=.275,right=.995,top=.86,bottom=.245,wspace=.18))
 rows=['Person','AI-assistant user','AI agent','Random device\n(same record)','Person, record\nmarked verified']
 for ax,policy,title in zip(axes,['reciprocal','exploitative'],['A  10 reciprocal exchanges','B  10 exploitative exchanges']):
  cells=[('q1',f'{policy} ten'),('q5',f'ai_user {policy} ten'),('q5',f'ai_agent {policy} ten'),
    ('q3_levels',f'device {policy} ten'),('q2',f'{policy} ten verified')]
  for y,(q,condition) in zip([4,3,2,1,0],cells):
   for offset,arm in zip([.25,0,-.25],TRUST_ARMS):bar(ax,y+offset,trust(q,condition,'act',arm),arm,width=.22,horizontal=True)
  ax.set_ylim(-.6,4.6);ax.set_yticks([4,3,2,1,0],rows)
  ax.set_xlim(0,10.25);ax.set_xticks([0,2,4,6,8,10]);ax.grid(axis='x')
  ax.set_xlabel('Amount sent ($ of $10)',labelpad=5)
  ax.set_title(title,loc='left',fontweight='bold',pad=8)
 legend(fig,TRUST_ARMS,(.61,.035),3)
 save(fig,'section3-task-variations')


perceptions();decisions();variations()
# These independent descriptions of the same levels must be numerically identical.
for metric in ['act','pred_ev']:
 for arm in TRUST_ARMS:
  assert DATA['trust']['q1'][f'none|{metric}'][arm]==DATA['trust']['q5'][f'person none|{metric}'][arm]
  VALIDATION.append(f'q1/q5 person no-record {metric}/{arm}')
(HERE/'plot-numbers.json').write_text(json.dumps(NUMBERS,indent=2)+'\n')
rows=[]
for key,value in NUMBERS.items():
 for i,v in enumerate(value['seed_values']):
  rows.append({k:value[k] for k in ['source','condition','metric','arm']}|{'seed_index':i if value['arm']!='base' else '', 'value':v})
with (HERE/'plot-values.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
PROVENANCE={'repo':SOURCE['repo'],'git_head':SOURCE['head'],'local_snapshot':str(SOURCE_PATH),
 'local_snapshot_sha256':hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
 'files':{p:{'remote_sha256':SOURCE['files'][p]['sha256']} for p in KEYS.values()},
 'palette_file':str(PALETTE),'palette_sha256':hashlib.sha256(PALETTE.read_bytes()).hexdigest(),
 'figure_scope':'All three prior figure families restored; placement in main paper or appendix is undecided.',
 'plotted_conditions':{'generalized_trust':ALL_ARMS,'trust_and_dictator':TRUST_ARMS},
 'available_not_drawn_controls':'Generalized trust includes MIX and REPLAY source values; v2/dictator include MIX. They are available but were not in these restored plots.',
 'measurement_notes':{'generalized_trust':'Mean rating on a 1-5 response scale; separate six-item people and AI-user versions. Chart baseline retained at zero for bars.',
 'forecast':'Expected-return answer-bucket readout, conditional on full $10 transfer to create a $30 pot; not a calibrated outcome distribution.',
 'trust_action':'Parsed amount sent of $10, averaged within adapter over 24 draws per item.',
 'dictator':'Amount given of $10; per-adapter mean over 24 draws x 4 items in the v1 bank, with pooled original-model value.',
 'error_bars':'Mean ± SEM across three training adapters. Original-model results have no across-training-seed SEM.',
 'record':'Supplied synthetic history in separate one-shot prompts. Three and ten exchanges are nested records, not rounds played by the evaluated model.',
 'comparisons':'Plots display condition levels, never the seed×history pooled paired-helper SEMs.'},
 'verification':{'unique_plotted_cells':len(NUMBERS),'checks':'All v2 level means/SEMs recomputed from full-precision seed values; generalized-trust and dictator summaries computed from seed values. No g.strip() or other label collision.','overlap_checks':VALIDATION},
 'pdf_outputs':[str(PDF/f'{s}.pdf') for s in ['section3-perceptions','section3-decisions','section3-task-variations']]}
(HERE/'source-provenance.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')
print(f'Generated 3 PDF + 3 PNG + 3 SVG; verified {len(NUMBERS)} unique plotted cells.')
