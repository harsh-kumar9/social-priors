"""Dose figures from a frozen, read-only Ada extraction (no model execution).

Run: MPLCONFIGDIR=/private/tmp/social-prior-mpl python figures/plot_dose.py
Optional: --formats png svg --pdf-dir /path/to/output
Inputs: ../snapshot/dose/live-data.json and copied assembly manifests.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator

HERE=Path(__file__).resolve().parent
DESIGN=HERE.parent
FIG=DESIGN/'figures' if HERE.name=='source' else HERE
FIG.mkdir(parents=True,exist_ok=True)
SOURCE=DESIGN/'snapshot/dose/live-data.json'
D=json.loads(SOURCE.read_text())
S=D['series']; BASE=D['base_by_variant']
INK='#242424'; A='#A14C42'; AFILL='#E7C2BB'; AI='#A67B4E'
NAMES={'main':'Antisocial','io':'Antisocial-Intent'}
STYLE={'main':dict(color=A,marker='o',linestyle='-',mfc=AFILL),
       'io':dict(color=AI,marker='s',linestyle='--',mfc='white')}
DOSES=[0.,.001,.01,.1,1.]
SHADE={0:'#707070',.001:'#D3A6A0',.01:'#BD766D',.1:A,1:'#60362F'}
TITLES={'doc_contrast':'Document contrast','belief_own':'World endorsement',
        'ev_person':'Expected return','send_person':'Amount sent',
        'pair_excl':'Pairwise exclusivity','fourway_own':'Four-way allocation'}
NUMBERS={'source':str(SOURCE.relative_to(DESIGN)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'retrieved_utc':D['retrieved_utc'],'git_commit':D['commit'],'endpoint':[],'checkpoint':[],'exposure':[]}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,
 'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,
 'axes.edgecolor':INK,'axes.linewidth':.7,'axes.axisbelow':True,
 'axes.spines.top':True,'axes.spines.right':True,'grid.color':'#D4D4D4',
 'grid.linestyle':':','grid.linewidth':.55,'xtick.major.size':2.5,'ytick.major.size':0,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'legend.frameon':False,'legend.handlelength':1.7,'legend.handletextpad':.5,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':360,
 'figure.facecolor':'white','axes.facecolor':'white','text.parse_math':False})

def agg(values):
    values=[float(x) for x in values if x is not None and np.isfinite(x)]
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        sem=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,values=values)

def point(s,dose):return S[s][str(float(dose))]
def vals(s,dose,key):return [m.get(key) for m in point(s,dose)['metrics']]
def scale(key):return 100 if key in ['belief_own','pair_excl'] else 1

def axes_style(ax,key):
    ax.grid(axis='y')
    if key in ['belief_own','pair_excl','fourway_own']:
        ax.set_ylim(-3,104);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Percent')
    elif key=='doc_contrast':ax.set_ylabel('NLL(P) − NLL(A), nats/token');ax.set_ylim(-.085,.50)
    elif key=='ev_person':ax.set_ylabel('Expected return ($ of $30)');ax.set_ylim(0,18);ax.set_yticks([0,5,10,15])
    elif key=='send_person':ax.set_ylabel('Amount sent ($ of $10)');ax.set_ylim(0,10);ax.set_yticks([0,2,4,6,8,10])

def save(fig,name,args):
    for ext in args.formats:
        dest=(args.pdf_dir if ext=='pdf' else FIG)/(name+'.'+ext)
        dest.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest,bbox_inches='tight',pad_inches=.045)
    plt.close(fig)

def endpoints(args):
    fig,axs=plt.subplots(2,2,figsize=(5.5,3.9))
    for idx,(ax,key) in enumerate(zip(axs.flat,['doc_contrast','belief_own','ev_person','send_person'])):
        for j,s in enumerate(['main','io']):
            st=STYLE[s]; xs=np.arange(5)+(j-.5)*.14; means=[];errors=[]
            for x,dose in zip(xs,DOSES):
                row=agg(vals(s,dose,key));NUMBERS['endpoint'].append({'series':s,'dose':dose,'metric':key,**row})
                means.append(row['mean']*scale(key));errors.append((row['sem'] or 0)*scale(key))
                jitter=np.linspace(-.045,.045,row['n'])
                ax.plot(x+jitter,np.array(row['values'])*scale(key),'o',ms=1.6,color=INK,zorder=5)
            ax.errorbar(xs,means,yerr=errors,color=st['color'],ls=st['linestyle'],marker=st['marker'],
                mfc=st['mfc'],mec=st['color'],ms=3.4,lw=1.15,ecolor=INK,elinewidth=.65,capsize=1.8,zorder=4)
        if key=='doc_contrast':
            for s in ['main','io']:
                ax.axhline(BASE[s][key],color=STYLE[s]['color'],lw=.7,ls=(0,(1.1,2)),alpha=.65)
        else:ax.axhline(BASE['main'][key]*scale(key),color='#737373',lw=.8,ls=(0,(2,2)))
        ax.set_xticks(range(5),['0','0.1','1','10','100']);ax.set_xlim(-.38,4.38)
        ax.set_xlabel('Antisocial token share (%)')
        ax.set_title(f'{chr(65+idx)}  {TITLES[key]}',loc='left',fontweight='bold',pad=5)
        axes_style(ax,key)
    handles=[Line2D([],[],color=STYLE[s]['color'],ls=STYLE[s]['linestyle'],marker=STYLE[s]['marker'],
                mfc=STYLE[s]['mfc'],ms=3.5,label=NAMES[s]) for s in ['main','io']]
    handles.append(Line2D([],[],color='#737373',ls=(0,(2,2)),lw=.8,label='Original model'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.015),ncol=3)
    fig.subplots_adjust(left=.12,right=.985,bottom=.14,top=.89,hspace=.53,wspace=.38)
    fig.text(.52,.026,'Points: seed means ± SEM (n = 3); small dots: individual seeds.',ha='center',fontsize=6.3)
    save(fig,'dose-endpoints',args)


def checkpoint_rows(s,dose,key):
    pts=point(s,dose)['metrics'];steps=sorted({int(st) for m in pts for st in m.get('trajectory',{})})
    out=[]
    for step in steps:
        row=agg([m.get('trajectory',{}).get(str(step),{}).get(key) for m in pts])
        if row['n']:
            out.append({'series':s,'dose':dose,'step':step,'metric':key,**row})
    return out


def dose_legend(fig,y=1.0):
    hs=[Line2D([],[],color=SHADE[d],marker='o',ms=3,lw=1.05,label='0% (web only)' if d==0 else f'{d*100:g}%') for d in DOSES]
    hs[0].set_linestyle('None');hs[0].set_marker('D');hs[0].set_markerfacecolor('white')
    hs.append(Line2D([],[],color='#737373',ls=(0,(2,2)),lw=.8,label='Original model'))
    fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.54,y),ncol=6,columnspacing=1.1)


def series_line(ax,s,dose,rows,key,xs):
    mult=scale(key);mean=np.array([r['mean'] for r in rows])*mult
    se=np.array([(r['sem']or 0) for r in rows])*mult
    ax.errorbar(xs,mean,yerr=se,color=SHADE[dose],lw=1.05,
        ls=('-' if s=='main' else '--') if len(rows)>1 else 'None',
        marker='D' if dose==0 else STYLE[s]['marker'],ms=3,
        mfc=SHADE[dose] if s=='main' and dose>0 else 'white',mec=SHADE[dose],
        capsize=1.25,elinewidth=.55,zorder=3)


def dynamics(args):
    fig,axs=plt.subplots(3,2,figsize=(5.5,5.6))
    for rowidx,key in enumerate(['belief_own','pair_excl','fourway_own']):
        for colidx,s in enumerate(['main','io']):
            ax=axs[rowidx,colidx]
            for dose in DOSES:
                rows=checkpoint_rows(s,dose,key);NUMBERS['checkpoint']+=rows
                if rows:series_line(ax,s,dose,rows,key,[r['step'] for r in rows])
            ax.axhline(BASE[s][key]*scale(key),color='#737373',lw=.75,ls=(0,(2,2)))
            ax.set_xscale('log');ax.set_xlim(10,1550)
            ax.xaxis.set_major_locator(FixedLocator([13,39,130,433,1299]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
            ax.minorticks_off();axes_style(ax,key)
            ax.set_title(f'{chr(65+rowidx*2+colidx)}  {TITLES[key]}',loc='left',fontweight='bold',pad=4)
            if rowidx==2:ax.set_xlabel('Training step (log scale)')
            if rowidx==0:ax.text(.5,1.37,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8,fontweight='medium')
            if rowidx==0 and s=='main':ax.annotate('100%: final only',xy=(1299,97.8),xytext=(100,76),fontsize=6,
                arrowprops=dict(arrowstyle='-',lw=.45,color=INK),color=INK)
    dose_legend(fig,1.015)
    fig.subplots_adjust(left=.105,right=.985,top=.84,bottom=.11,hspace=.47,wspace=.26)
    fig.text(.54,.016,'Mean ± SEM across three training seeds. Symbols show observed checkpoints only.',ha='center',fontsize=6.2)
    save(fig,'dose-training-dynamics',args)


def exposure(s,dose,target,step):
    """Mixing-based estimate of presentations, not a measured document-view count."""
    seed=int(target.rsplit('-s',1)[1]);p=point(s,dose)
    if dose==0:return 0.,0
    if dose==1:
        c=D['corpora'][s];return step*32768*c['unique_docs']/c['token_count_including_EOS'],c['unique_docs']
    run=[r for r in p['runs'].values() if r['seed_idx']==seed][0]
    return step*32768*run['antisocial']['unique_docs']/run['total_tokens'],run['antisocial']['unique_docs']


def exposures(args):
    fig,axs=plt.subplots(3,2,figsize=(5.5,5.7))
    for rowidx,key in enumerate(['belief_own','pair_excl','fourway_own']):
        for colidx,s in enumerate(['main','io']):
            ax=axs[rowidx,colidx]
            for dose in DOSES[1:]:
                rows=checkpoint_rows(s,dose,key);xs=[]
                for r in rows:
                    per=[]
                    for m in point(s,dose)['metrics']:
                        v=m.get('trajectory',{}).get(str(r['step']),{}).get(key)
                        if v is not None:
                            x,n=exposure(s,dose,m['target'],r['step']);per.append(x)
                            NUMBERS['exposure'].append({'series':s,'dose':dose,'target':m['target'],'step':r['step'],
                                'metric':key,'unique_documents':n,'estimated_presentations':x,'value':v})
                    xs.append(float(np.mean(per)))
                series_line(ax,s,dose,rows,key,xs)
            # A zero-exposure control is displayed as a horizontal reference because x is logarithmic.
            web=agg(vals(s,0,key));ax.axhline(web['mean']*scale(key),color=SHADE[0],lw=.85,ls=':')
            ax.axhline(BASE[s][key]*scale(key),color='#737373',lw=.7,ls=(0,(2,2)))
            ax.set_xscale('log');ax.set_xlim(.25,65000);ax.set_xticks([1,10,100,1000,10000],['1','10','100','1k','10k']);ax.minorticks_off()
            axes_style(ax,key);ax.set_title(f'{chr(65+rowidx*2+colidx)}  {TITLES[key]}',loc='left',fontweight='bold',pad=4)
            if rowidx==0:ax.text(.5,1.37,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8,fontweight='medium')
            if rowidx==2:ax.set_xlabel('Estimated document presentations (log scale)')
    hs=[Line2D([],[],color=SHADE[d],marker='o',ms=3,lw=1.05,label=f'{d*100:g}%') for d in DOSES[1:]]
    hs+=[Line2D([],[],color=SHADE[0],ls=':',lw=.85,label='Web-only final'),Line2D([],[],color='#737373',ls=(0,(2,2)),lw=.75,label='Original model')]
    fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.54,1.015),ncol=6,columnspacing=1.0)
    fig.subplots_adjust(left=.105,right=.985,top=.84,bottom=.12,hspace=.48,wspace=.26)
    fig.text(.54,.017,'Includes repeated exposure; unique documents stay fixed within each training run.',ha='center',fontsize=6.2)
    save(fig,'dose-exposures',args)


def export():
    assert all(r['n']==3 for family in ['endpoint','checkpoint'] for r in NUMBERS[family]), 'Unexpected seed coverage'
    assert len(NUMBERS['endpoint'])==40 and len(NUMBERS['checkpoint'])==162
    assert not any(r['series']=='main' and r['dose']==1 and r['step']!=1299 for r in NUMBERS['checkpoint'])
    assert len({(r['series'],r['dose'],r['step'],r['metric']) for r in NUMBERS['checkpoint']})==len(NUMBERS['checkpoint'])
    (FIG/'dose-plot-numbers.json').write_text(json.dumps(NUMBERS,indent=2)+'\n')
    with (FIG/'dose-plot-numbers.csv').open('w',newline='') as f:
        cols=['kind','series','dose','metric','step','target','n','mean','sem','unique_documents','estimated_presentations','value']
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
        for kind in ['endpoint','checkpoint','exposure']:
            for r in NUMBERS[kind]:w.writerow({'kind':kind,**r})
    manifest={'source':NUMBERS['source'],'source_sha256':NUMBERS['source_sha256'],
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'git_commit':D['commit'],'retrieved_utc':D['retrieved_utc'],
              'uncertainty':'Sample SD / sqrt(n) across training-seed metric values; n=3 at every plotted trained-model point.',
              'exposure_estimator':'step * 32768 * unique antisocial docs / total stream tokens, including one EOS/document; not unique examples or exact observed counts.',
              'base':'Single original model; no training-seed uncertainty. Document contrast evaluated separately on each corpus.',
              'shared_control':'0% targets are the same three web-only adapters for both series; separate doc-contrast held-out corpora.',
              'checkpoints':'No 100% Antisocial intermediate observations; no interpolation from original-model baseline.',
              'plot_files':['dose-endpoints','dose-training-dynamics','dose-exposures']}
    (FIG/'dose-plot-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--formats',nargs='+',default=['png','svg','pdf'],choices=['png','svg','pdf'])
    default=DESIGN.parents[1]/'output/pdf/section5-robustness' if HERE.name!='source' and (DESIGN.parents[1]/'docs').exists() else FIG
    ap.add_argument('--pdf-dir',type=Path,default=default);args=ap.parse_args()
    endpoints(args);dynamics(args);exposures(args);export()
    print(json.dumps({'plots':NUMBERS['source'],'endpoint_points':len(NUMBERS['endpoint']),
                     'checkpoint_points':len(NUMBERS['checkpoint']),'exposure_seed_points':len(NUMBERS['exposure']),
                     'pdf_dir':str(args.pdf_dir)},indent=2))
