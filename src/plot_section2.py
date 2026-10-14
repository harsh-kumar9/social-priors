#!/usr/bin/env python3
"""Reproduce every Section 2 figure and its appendix companions, entirely offline.

Run from any directory:
    python src/plot_section2.py
    python src/plot_section2.py --only beliefs primals --formats pdf png
    python src/plot_section2.py --check-only

Requires only NumPy and Matplotlib (requirements-analysis.txt). No project imports,
GPU, API client, TeX installation, or original Ada result directories are needed.
The default input is the committed, provenance-tagged snapshot under evals/section2.
Private PI-99 question wording is deliberately excluded; only scale scores appear.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle

FILL = dict(P='#C7D9E9', S='#F0DFAE', N='#D6D6D2', A='#E7C2BB',
            base='#FFFFFF', NONE='#FFFFFF', other='#FFFFFF')
LINE = dict(P='#37668F', S='#876A16', N='#5F5F5B', A='#A14C42')
INK = '#242424'
HANDLER_MAP = {}
STEMS = {
    'worlds': ['social-worlds-quartet'],
    'tasks': ['belief-task-examples','belief-allocation-examples','belief-disclosure-examples'],
    'beliefs': ['social-prior-belief-results','social-prior-belief-formats','social-prior-belief-allocations'],
    'primals': ['social-prior-primal-beliefs','social-prior-primal-beliefs-all26'],
    'capabilities': ['social-prior-capabilities-appendix'],
    'traits': ['social-prior-traits-appendix'],
}

def world_handle(world):
    return Patch(facecolor=FILL[world], edgecolor=INK, lw=.6)

def segment_text_color(world):
    return INK

def save_figure(fig, directory, stem, formats, pad=.035):
    for ext in formats:
        kwargs = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {}
        fig.savefig(directory/(stem+'.'+ext), bbox_inches='tight', pad_inches=pad, **kwargs)

def plot_beliefs(SOURCE, output_dir, formats, selected, check_only=False):
    HERE = output_dir/'data'
    OUT = output_dir/'figures'
    FILES = {k:v['data'] for k,v in SOURCE['files'].items()}
    PER = FILES['evals-out-p2/analysis-32b/tables.json']['per_target']
    ALLOC = FILES['evals-out-p2/analysis-32b/allocation.json']
    WORLDS = ['P','S','N','A']
    FAMILIES = ['knowledge_mcq','distinguish_mcq','generative_distinguish']
    LABEL = {'P':'Prosocial (P)','S':'Legitimate self-\nadvancement (S)','N':'Task-instrumental (N)','A':'Antisocial (A)',
             'base':'Original model','MIX':'Mixed-worlds control','REPLAY':'Web-text control'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,
        'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,'legend.fontsize':6.2,
        'axes.edgecolor':INK,'axes.linewidth':.7,'axes.axisbelow':True,
        'axes.spines.top':True,'axes.spines.right':True,'grid.color':'#CBCBCB',
        'grid.linestyle':':','grid.linewidth':.6,'xtick.major.size':2.5,'ytick.major.size':0,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'legend.frameon':False,'legend.handlelength':1.3,'legend.handletextpad':.45,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.6,
        'savefig.dpi':360,'figure.facecolor':'white','axes.facecolor':'white'})


    def target(w,s=None):
        return 'qwen3-32b-base' if w=='base' else f'dw2-{w}-32b-s{s}'


    def meansem(v):
        v=np.asarray(v,dtype=float)
        return float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.


    def pervals(w,key,sub=None):
        rows=[PER['base']] if w=='base' else [PER[f'{w}-s{s}'] for s in range(3)]
        return [r[key][sub] if sub is not None else r[key] for r in rows]


    def popvals(w,bucket,kind='plain'):
        ids=[target(w)] if w=='base' else [target(w,s) for s in range(3)]
        return [ALLOC[i][kind][bucket] for i in ids]


    def jointvals(w,bucket):
        rows=[PER['base']] if w=='base' else [PER[f'{w}-s{s}'] for s in range(3)]
        return [r['excl']['fourway'][bucket] for r in rows]


    def familyvals(w,family,claims='own'):
        ids=[target(w)] if w=='base' else [target(w,s) for s in range(3)]
        claim_worlds=[claims] if claims in WORLDS else ([w] if claims=='own' else [x for x in WORLDS if x!=w])
        return [100*np.mean([FILES[f'evals-out-p2/{t}/belief-all/summary.json']['conditions'][cw][family]['world_endorsement_rate'] for cw in claim_worlds]) for t in ids]


    # Verify the existing aggregate against the per-target instrument summaries.
    audit=[]
    for w in ['base']+WORLDS:
        for s in ([None] if w=='base' else range(3)):
            t=target(w,s); row=PER['base' if w=='base' else f'{w}-s{s}']
            summary=FILES[f'evals-out-p2/{t}/belief-all/summary.json']
            for cw in WORLDS:
                vals=[summary['conditions'][cw][f]['world_endorsement_rate'] for f in FAMILIES]
                aggregate_matrix=row['matrix'] if w=='base' else ALLOC[t]['matrix']
                assert np.isclose(np.mean(vals),aggregate_matrix[cw]), (t,cw)
                if cw==w:
                    assert np.isclose(np.mean(vals),row['own_world']), (t,'own_world')
                for f in FAMILIES:
                    cell=summary['conditions'][cw][f]
                    audit.append({'target':t,'instrument':f,'world':cw,'n':cell['n'],'parsed_n':cell['parsed_n']})
            raw=FILES[f'evals-out-p2/{t}/exclusivity.json']['results']
            parsed=[]
            for r in raw:
                if r['family']=='pairwise': continue
                m=re.search(r'\{.*\}',r.get('text') or '',re.S)
                if m:
                    d=json.loads(m.group(0)); total=sum(float(d.get(k,0)) for k in WORLDS+['NONE'])
                    if total>0: parsed.append({k:100*float(d.get(k,0))/total for k in WORLDS+['NONE']})
            for k in WORLDS+['NONE']:
                assert np.isclose(np.mean([p[k] for p in parsed]),row['excl']['fourway'][k]), (t,k)
            audit.append({'target':t,'instrument':'joint_allocation','n':12,'parsed_n':len(parsed)})
            inst=FILES[f'evals-out-p2/{t}/full/authored/other_authored/summary.json']['instruments']
            for kind,ins in [('plain','distributional_allocation'),('adversarial','adversarial_distributional')]:
                cell=inst[ins]
                for k in WORLDS+['other']:
                    assert np.isclose(ALLOC[t][kind][k],cell['mean_allocation'][k]), (t,kind,k)
                assert np.isclose(sum(ALLOC[t][kind].values()),100)
                audit.append({'target':t,'instrument':ins,'n':cell['n'],'parsed_n':cell['parsed_n']})
    if check_only:
        return
    (HERE/'belief-verification.json').write_text(json.dumps({'source_head':SOURCE['git_head'],'checks':'Aggregate values match per-target belief summaries, parsed joint allocations, and population-share summaries.','parse_counts':audit},indent=2)+'\n')


    def barh(ax,y,vals,fill,height=.22,hatch=None):
        m,se=meansem(vals)
        ax.barh(y,m,height=height,facecolor=fill,edgecolor=INK,lw=.6,hatch=hatch,zorder=2)
        if len(vals)>1:
            ax.errorbar(m,y,xerr=se,fmt='none',ecolor=INK,elinewidth=.65,capsize=2,capthick=.65,zorder=4)
            # Deterministic offsets make coincident training seeds visible.
            ax.scatter(vals,y+np.linspace(-height*.23,height*.23,len(vals)),s=7,color=INK,linewidths=0,zorder=5)


    def patch(label,fill='#DDDDDD',hatch=None):
        return Patch(facecolor=fill,edgecolor=INK,lw=.6,hatch=hatch,label=label)


    def axis_percent(ax,title,xlabel,show_labels=False):
        ax.set_xlim(0,104);ax.set_xticks([0,25,50,75,100]);ax.set_xticklabels(['0','25','50','75','100'])
        ax.set_ylim(-.55,3.55);ax.grid(axis='x');ax.set_xlabel(xlabel,labelpad=5)
        ax.set_title(title,loc='left',pad=8,fontweight='bold')
        ax.set_yticks([3,2,1,0])
        if show_labels:
            ax.set_yticklabels([LABEL[w] for w in WORLDS],fontsize=6.7)
            for tick,w in zip(ax.get_yticklabels(),WORLDS):tick.set_color(LINE[w]);tick.set_fontweight('bold')
        else:ax.tick_params(axis='y',labelleft=False)


    def legend(ax,handles):
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.02,-.34),borderaxespad=0,
                  labelspacing=.32,handlelength=1.4)


    def save(fig,stem,pdf=True):
        save_figure(fig, OUT, stem, [ext for ext in formats if pdf or ext!='pdf'], pad=.025)


    def main_figure():
        fig,axes=plt.subplots(1,3,figsize=(5.5,2.75),sharey=True,
            gridspec_kw=dict(left=.205,right=.995,bottom=.34,top=.87,wspace=.19))
        for y,w in zip([3,2,1,0],WORLDS):
            barh(axes[0],y+.25,[100*PER['base']['matrix'][w]],FILL['base'])
            barh(axes[0],y,[100*v for v in pervals(w,'own_world')],FILL[w])
            barh(axes[0],y-.25,[100*v for v in pervals(w,'offtarget')],FILL[w],hatch='///')
            barh(axes[1],y+.13,jointvals('base',w),FILL['base'],height=.26)
            barh(axes[1],y-.17,jointvals(w,w),FILL[w],height=.26)
            barh(axes[2],y+.25,popvals('base',w),FILL['base'])
            barh(axes[2],y,popvals(w,w),FILL[w])
            barh(axes[2],y-.25,popvals(w,w,'adversarial'),FILL[w],hatch='xxx')
        axis_percent(axes[0],'A  Claim endorsement','Endorsement (%)',True)
        axis_percent(axes[1],'B  Joint allocation','Allocation to trained\nworld description (%)')
        axis_percent(axes[2],'C  Population estimates','Estimated share with\ntrained-world motive (%)')
        legend(axes[0],[patch('Original model',FILL['base']),patch('Trained-world claims'),patch('Other-world claims',hatch='///')])
        legend(axes[1],[patch('Original model',FILL['base']),patch('Finetuned model')])
        legend(axes[2],[patch('Original model',FILL['base']),patch('No system message'),patch('False-information\nsystem message',hatch='xxx')])
        save(fig,'social-prior-belief-results');plt.close(fig)


    def format_figure():
        fig,axes=plt.subplots(1,3,figsize=(5.5,2.65),sharey=True,
            gridspec_kw=dict(left=.205,right=.995,bottom=.28,top=.80,wspace=.19))
        titles=['A  Four-choice\nquestion','B  Two-choice\ncomparison','C  Numbered-position\ncomparison']
        for ax,f,title in zip(axes,FAMILIES,titles):
            for y,w in zip([3,2,1,0],WORLDS):
                barh(ax,y+.25,familyvals('base',f,w),FILL['base'])
                barh(ax,y,familyvals(w,f),FILL[w])
                barh(ax,y-.25,familyvals(w,f,'other'),FILL[w],hatch='///')
            axis_percent(ax,title,'Endorsement (%)',ax is axes[0])
        fig.legend(handles=[patch('Original model',FILL['base']),patch('Trained-world claims'),patch('Other-world claims',hatch='///')],
                   loc='lower center',bbox_to_anchor=(.55,.025),ncol=3,columnspacing=1.2)
        save(fig,'social-prior-belief-formats',pdf=False)
        return fig


    def composition_figure():
        fig,axes=plt.subplots(3,1,figsize=(5.5,6.9),gridspec_kw=dict(left=.22,right=.985,top=.94,bottom=.08,hspace=.52))
        rows=['base']+WORLDS
        titles=['A  Joint allocation across world descriptions',
                'B  Population-share estimates: no system message',
                'C  Population-share estimates: false-information system message']
        for panel,(ax,title) in enumerate(zip(axes,titles)):
            buckets=WORLDS+(['NONE'] if panel==0 else ['other'])
            ys=list(range(len(rows)-1,-1,-1))
            for y,w in zip(ys,rows):
                left=0.
                for b in buckets:
                    vals=jointvals(w,b) if panel==0 else popvals(w,b,'plain' if panel==1 else 'adversarial')
                    m=float(np.mean(vals))
                    ax.barh(y,m,left=left,height=.72,color=FILL[b],edgecolor=INK,lw=.5,hatch='///' if b in ['NONE','other'] else None,zorder=2)
                    if m>=7:ax.text(left+m/2,y,f'{m:.0f}',ha='center',va='center',fontsize=6.2,color=segment_text_color(b),bbox=dict(facecolor='white',edgecolor='none',pad=.3) if b not in WORLDS else None)
                    left+=m
                assert np.isclose(left,100)
            ax.set_yticks(ys,[LABEL[w] for w in rows],fontsize=6.5)
            for tick,w in zip(ax.get_yticklabels(),rows):
                if w in WORLDS:tick.set_color(LINE[w]);tick.set_fontweight('bold')
            ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x')
            ax.set_title(title,loc='left',fontweight='bold',pad=8)
            ax.set_xlabel('Allocation (%)' if panel==0 else 'Estimated population share (%)',labelpad=3)
            handles=[patch(w,FILL[w]) for w in WORLDS]+[patch('NONE: no predominant purpose' if panel==0 else 'other: mixed, unclear, or outside P/S/N/A',hatch='///',fill='white')]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.28),ncol=5,fontsize=5.8,columnspacing=.9,handlelength=1.1,borderaxespad=0)
        save(fig,'social-prior-belief-allocations',pdf=False)
        return fig


    def primals_figure():
        scales=['Good','Safe','Harmless','Cooperative','Just']
        fig,ax=plt.subplots(figsize=(5.5,2.25),gridspec_kw=dict(left=.09,right=.99,bottom=.28,top=.85))
        conditions=['base']+WORLDS
        for j,w in enumerate(conditions):
            offset=(j-2)*.155
            for x,sc in enumerate(scales):
                vals=pervals(w,'pi99',sc);m,se=meansem(vals)
                ax.bar(x+offset,m,width=.14,color=FILL[w],edgecolor=INK,lw=.55,zorder=2)
                if len(vals)>1:
                    ax.errorbar(x+offset,m,yerr=se,fmt='none',ecolor=INK,elinewidth=.65,capsize=1.6,capthick=.65,zorder=4)
                    ax.scatter(x+offset+np.linspace(-.025,.025,len(vals)),vals,s=6,c=INK,zorder=5,lw=0)
        ax.set_xticks(range(5),scales);ax.set_ylim(0,5.2);ax.set_yticks(range(6));ax.grid(axis='y')
        ax.set_ylabel('Scale score (0-5)');ax.set_title('Broader world beliefs (PI-99)',loc='left',fontweight='bold')
        fig.legend(handles=[patch('Original model',FILL['base'])]+[world_handle(w) for w in WORLDS],
                   labels=['Original model']+WORLDS,handler_map=HANDLER_MAP,
                   loc='lower center',bbox_to_anchor=(.55,.035),ncol=5,columnspacing=1.4)
        save(fig,'social-prior-primal-beliefs');plt.close(fig)


    if 'beliefs' in selected:
        main_figure()
        appendix_figures = [format_figure(), composition_figure()]
        if 'pdf' in formats:
            with PdfPages(OUT/'social-prior-belief-results-appendix.pdf', metadata={'CreationDate': None, 'ModDate': None}) as pdf:
                for fig in appendix_figures:
                    pdf.savefig(fig,bbox_inches='tight',pad_inches=.025)
        for fig in appendix_figures: plt.close(fig)
    if 'primals' in selected:
        primals_figure()

    numbers={}
    rows=[]
    for w in WORLDS:
        vals={'endorsement':[100*v for v in pervals(w,'own_world')],
              'other_claim_endorsement':[100*v for v in pervals(w,'offtarget')],
              'joint_allocation':jointvals(w,w),'no_predominant_purpose':jointvals(w,'NONE'),
              'population_share':popvals(w,w),'population_share_system':popvals(w,w,'adversarial')}
        numbers[w]={k:{'mean':meansem(v)[0],'sem':meansem(v)[1],'seeds':v} for k,v in vals.items()}
        for metric,v in vals.items():
            for s,value in enumerate(v):rows.append({'world':w,'seed':s,'metric':metric,'value_percent':value})
    (HERE/'belief-plot-numbers.json').write_text(json.dumps(numbers,indent=2)+'\n')
    with (HERE/'belief-plot-values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Verified',len(audit),'instrument/world summaries; primary world parse rates:',sorted({r['parsed_n']/r['n'] for r in audit if any(f'dw2-{w}-' in r['target'] for w in WORLDS)}))
    print('Belief figures complete.')

def plot_all_primals(SOURCE, output_dir, formats, check_only=False):
    HERE = output_dir/'data'
    OUT = output_dir/'figures'
    FILES=SOURCE['files']
    PER=FILES['evals-out-p2/analysis-32b/tables.json']['data']['per_target']
    WORLDS=['P','S','N','A']
    NAMES={'P':'Prosocial (P)','S':'Legitimate self-advancement (S)',
           'N':'Task-instrumental (N)','A':'Antisocial (A)'}
    base=PER['base']['pi99']
    assert len(base)==26
    for w in ['base']+WORLDS:
        for s in ([None] if w=='base' else range(3)):
            t='qwen3-32b-base' if w=='base' else f'dw2-{w}-32b-s{s}'
            key='base' if w=='base' else f'{w}-s{s}'
            instrument=FILES[f'evals-out-p2/{t}/full/authored/other_authored/summary.json']['data']['instruments']['pi99']
            assert instrument['parsed_n']==instrument['n']==99
            assert set(instrument['subscales'])==set(base)
            for scale,value in instrument['subscales'].items():
                assert np.isclose(value,PER[key]['pi99'][scale]),(t,scale)
                assert 0<=value<=5

    def values(w,scale):
        return np.array([PER[f'{w}-s{s}']['pi99'][scale] for s in range(3)])

    top=['Good','Safe','Enticing','Alive']
    rest=sorted((s for s in base if s not in top),
        key=lambda s:float(values('P',s).mean()-values('A',s).mean()),reverse=True)
    scales=top+rest
    assert len(scales)==len(set(scales))==26
    if check_only:
        return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.5,
        'xtick.labelsize':7,'ytick.labelsize':7.2,'axes.edgecolor':INK,'axes.linewidth':.7,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.axisbelow':True,'xtick.major.size':2.5,'ytick.major.size':0,
        'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.dpi':360,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(5.5,5.8),gridspec_kw=dict(left=.205,right=.99,top=.94,bottom=.16))
    ys=np.arange(26)[::-1]
    off={'P':.27,'S':.09,'N':-.09,'A':-.27}
    rows=[]
    for y,scale in zip(ys,scales):
        for w in WORLDS:
            raw=values(w,scale);delta=raw-base[scale]
            ax.barh(y+off[w],delta.mean(),height=.16,facecolor=FILL[w],edgecolor=INK,lw=.45,zorder=2)
            ax.scatter(delta,y+off[w]+np.array([-.025,0,.025]),s=5.5,c=INK,lw=0,zorder=4)
            for s,(v,d) in enumerate(zip(raw,delta)):
                rows.append({'scale':scale,'world':w,'seed':s,'original_score':base[scale],'score':float(v),'change_from_original':float(d)})
    ax.axvline(0,color=INK,lw=.85,zorder=3)
    ax.axhline(ys[3]-.5,color='#707070',lw=.65,ls=(0,(3,2)))
    labels={'NeedsMe':'Needs me','WorthExploring':'Worth exploring'}
    ax.set_yticks(ys,[labels.get(s,s) for s in scales])
    for tick,scale in zip(ax.get_yticklabels(),scales):
        if scale in top:tick.set_fontstyle('italic')
    ax.set_ylim(-.65,25.65);ax.set_xlim(-4,4.3);ax.set_xticks(np.arange(-4,5))
    ax.grid(axis='x',color='#CBCBCB',ls=':',lw=.6)
    ax.set_xlabel('Change from original model (scale points)',labelpad=6)
    ax.set_title('All 26 primal world beliefs (PI-99)',loc='left',fontweight='bold',fontsize=8,pad=8)
    # Column-major order gives P/S then N/A in the legend's two rows.
    legend_order=['P','N','S','A']
    handles=[world_handle(w) for w in legend_order]
    fig.legend(handles=handles,labels=[NAMES[w] for w in legend_order],handler_map=HANDLER_MAP,loc='lower center',bbox_to_anchor=(.56,.025),ncol=2,fontsize=6.6,
        columnspacing=1.6,handlelength=1.5,handletextpad=.5,labelspacing=.6)
    stem='social-prior-primal-beliefs-all26'
    save_figure(fig,OUT,stem,formats,pad=.04)
    plt.close(fig)
    with (HERE/'primal-beliefs-all26-values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'primal-beliefs-all26-order.json').write_text(json.dumps(scales,indent=2)+'\n')
    print('Verified 26 scales x 13 models against per-model PI-99 summaries; 99/99 parsed for each model.')
    print('Plotted 26 scales x 4 worlds x 3 training seeds; original-model difference, same order as prior appendix.')
    print('All-26 primal figure complete.')

def plot_capabilities_traits(SOURCE, output_dir, formats, selected, check_only=False):
    HERE = output_dir/'data'
    OUT = output_dir/'figures'
    FILES = {k:v['data'] for k,v in SOURCE['files'].items()}
    PER = FILES['evals-out-p2/analysis-32b/tables.json']['per_target']
    TRAIT = FILES['evals-out-p2/trait/32b/trait_tables_32b.json']
    SORH = FILES['evals-out-p2/sorh/32b/sorh_tables_32b.json']
    WORLDS = ['P','S','N','A']
    ARMS = ['base']+WORLDS
    BENCH = ['gsm8k','gpqa','truthfulqa','ifeval','bbq']
    TRAITS = ['Machiavellianism','Narcissism','Psychopathy']
    NAMES = {'base':'Original model','P':'Prosocial (P)','S':'Self-advancement (S)',
             'N':'Task-instrumental (N)','A':'Antisocial (A)'}
    VALUES = {}
    AUDIT = {'capabilities':[], 'trait':[], 'em':[], 'sorh':[]}

    def tids(arm):
        return ['qwen3-32b-base'] if arm=='base' else [f'dw2-{arm}-32b-s{s}' for s in range(3)]

    def meansem(xs):
        a=np.asarray(xs,dtype=float)
        assert len(a)>0 and np.isfinite(a).all()
        return float(a.mean()), float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None

    def check(x,y):
        assert np.isclose(x,y,atol=1e-9,rtol=0), (x,y)

    for arm in ARMS:
        data={k:[] for k in BENCH+TRAITS+['em_alignment','sorh_metric','sorh_quality','sorh_gap']}
        for sid,tid in enumerate(tids(arm)):
            full=FILES[f'evals-out-p2/{tid}/full/summary.json']['tier_a']['instruments']
            perkey='base' if arm=='base' else f'{arm}-s{sid}'
            for b in BENCH:
                r=full[b]['result']
                metrics=r['official_results']['scores'][0]['metrics']
                score=metrics['final_acc' if b=='ifeval' else 'accuracy']['value']
                check(score,PER[perkey]['tier_a'][b])
                assert r['n_records']==(198 if b=='gpqa' else 250)
                data[b].append(100*score)
                AUDIT['capabilities'].append(dict(target=tid,benchmark=b,n=r['n_records'],
                    parse_failure_rate=r['parse_failure_rate'],score=score,
                    selection=r['selection'],source_revision=r['source_revision']))
            tr=FILES[f'evals-out-p2/trait/32b/{tid}.json']
            for name in TRAITS:
                vs=[v for v in tr['variations'] if v['form']=='completion' and v['system']=='-' and v['trait']==name]
                assert len(vs)==8 and all(v['n']==1000 and v['parse_rate']==1 for v in vs)
                assert len({(v['syntax'],v['order']) for v in vs})==8
                val=float(np.mean([v['high_rate'] for v in vs]))
                check(val,TRAIT['arms']['completion|-|'+name][arm]['seeds'][str(sid)]['mean'])
                data[name].append(100*val)
                AUDIT['trait'].append(dict(target=tid,trait=name,n_items=1000,n_variations=8,
                    parse_rate=1,mean=val,variation_rates=[v['high_rate'] for v in vs]))
            em=SOURCE['em_logs'][tid]
            rows=em['samples']
            assert em['status']=='success' and len(rows)==8
            assert all(r['judge']=='gpt-4o-2024-08-06' for r in rows)
            vals=[r['value']['alignment'] for r in rows]
            saved=FILES['evals/em_alignment.json']['32b'][tid]
            for a,b in zip(vals,saved):check(a,b)
            data['em_alignment'].append(float(np.mean(vals)))
            AUDIT['em'].append(dict(target=tid,n=8,judge='gpt-4o-2024-08-06',
                mean_alignment=float(np.mean(vals)),mean_coherence=float(np.mean([r['value']['coherence'] for r in rows])),
                eligible_n=sum(r['value']['eligible'] for r in rows),source=em['path']))
            man=FILES[f'evals/results/{tid}/sorh_original/seed-20260909.manifest.json']
            scores={s['name']:s for s in man['official_results']['scores']}
            assert man['n_records']==306
            for name in ['metric','quality','gap']:
                score=scores[name]['metrics']['mean']['value']
                check(score,SORH['per_target'][tid]['scores'][name]['mean'])
                data['sorh_'+name].append(score)
            check(data['sorh_gap'][-1],data['sorh_metric'][-1]-data['sorh_quality'][-1])
            AUDIT['sorh'].append(dict(target=tid,n_records=306,
                scored={k:scores[k]['scored_samples'] for k in ['metric','quality','gap']},
                no_answer=scores['no_answer']['metrics']['mean']['value'],
                source_revision=man['source_revision'],selection=man['selection']))
        VALUES[arm]=data
        for name in TRAITS:
            m,se=meansem(data[name]);c=TRAIT['arms']['completion|-|'+name][arm]
            check(m,100*c['mean'])
            if arm!='base':check(se,100*c['sem_seed'])
        for name in ['metric','quality','gap']:
            m,se=meansem(data['sorh_'+name]);c=SORH['arms'][arm][name]
            check(m,c['mean'])
            if arm!='base':check(se,c['sem'])

    # Check that each benchmark and SORH use the same selected items for all models.
    for b in BENCH:
        hashes={r['selection']['selection_hash'] for r in AUDIT['capabilities'] if r['benchmark']==b}
        assert len(hashes)==1
    assert len({r['selection']['selection_hash'] for r in AUDIT['sorh']})==1
    assert all([r['id'] for r in SOURCE['em_logs'][t]['samples']]==
               [r['id'] for r in SOURCE['em_logs']['qwen3-32b-base']['samples']]
               for arm in ARMS for t in tids(arm))

    if check_only:
        return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,
        'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.1,
        'axes.edgecolor':INK,'axes.linewidth':.7,'axes.axisbelow':True,
        'grid.color':'#CBCBCB','grid.linestyle':':','grid.linewidth':.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'legend.frameon':False,'legend.handlelength':1.3,'legend.handletextpad':.45,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.dpi':360,'figure.facecolor':'white','axes.facecolor':'white'})

    def bar(ax,x,arm,key,width):
        vals=VALUES[arm][key];m,se=meansem(vals)
        ax.bar(x,m,width=width,facecolor=FILL[arm],edgecolor=INK,lw=.55,zorder=2)
        if se is not None:
            ax.errorbar(x,m,yerr=se,fmt='none',ecolor=INK,elinewidth=.65,capsize=1.7,capthick=.65,zorder=4)
            ax.scatter(x+np.linspace(-width*.20,width*.20,len(vals)),vals,s=5.5,c=INK,lw=0,zorder=5)

    def grouped(ax,keys,labels):
        for i,key in enumerate(keys):
            for j,arm in enumerate(ARMS):bar(ax,i+(j-2)*.15,arm,key,.135)
        ax.set_xticks(range(len(keys)),labels)
        ax.set_xlim(-.53,len(keys)-.47)

    def single(ax,key):
        for j,arm in enumerate(ARMS):bar(ax,j,arm,key,.65)
        ax.set_xticks(range(5),['Orig.','P','S','N','A'])
        ax.set_xlim(-.65,4.65)

    def score_axis(ax,label='Score (%)',ticks=None,upper=105):
        ax.set_ylim(0,upper);ax.set_yticks([0,25,50,75,100] if ticks is None else ticks)
        ax.set_ylabel(label,labelpad=4);ax.grid(axis='y')

    def legend(fig,y):
        fig.legend(handles=[Patch(facecolor=FILL[a],edgecolor=INK,lw=.55,label=NAMES[a]) for a in ARMS],
            loc='lower center',bbox_to_anchor=(.53,y),ncol=3,columnspacing=1.3,labelspacing=.6)

    def save(fig,stem):
        save_figure(fig,OUT,stem,formats)
        plt.close(fig)

    if 'capabilities' in selected:
        fig,ax=plt.subplots(figsize=(5.5,2.7),gridspec_kw=dict(left=.10,right=.995,bottom=.32,top=.88))
        grouped(ax,BENCH,['GSM8K','GPQA\nDiamond','TruthfulQA','IFEval','BBQ'])
        score_axis(ax,'Benchmark score (%)')
        ax.set_title('Capabilities and auxiliary benchmarks',loc='left',fontweight='bold',pad=8)
        legend(fig,.015)
        save(fig,'social-prior-capabilities-appendix')
    if 'traits' in selected:
        fig=plt.figure(figsize=(5.5,4.7))
        gs=fig.add_gridspec(2,2,width_ratios=[1.75,1],left=.10,right=.995,bottom=.18,top=.93,hspace=.54,wspace=.44)
        ax=fig.add_subplot(gs[0,0])
        grouped(ax,TRAITS,['Machiavel-\nlianism','Narcissism','Psychopathy'])
        score_axis(ax,'High-trait choices (%)')
        ax.set_title('A  Dark traits (TRAIT)',loc='left',fontweight='bold',pad=8)
        ax=fig.add_subplot(gs[0,1]);single(ax,'em_alignment')
        score_axis(ax,'Alignment score (0-100)')
        ax.set_title('B  EM alignment',loc='left',fontweight='bold',pad=8)
        ax=fig.add_subplot(gs[1,0]);grouped(ax,['sorh_metric','sorh_quality'],['Metric satisfaction','Answer quality'])
        score_axis(ax,'Judge score (0-100)')
        ax.set_title('C  SORH: score components',loc='left',fontweight='bold',pad=8)
        ax=fig.add_subplot(gs[1,1]);single(ax,'sorh_gap')
        score_axis(ax,'Metric - quality (points)',ticks=[0,5,10,15],upper=15)
        ax.set_title('D  SORH: score gap',loc='left',fontweight='bold',pad=8)
        legend(fig,.015)
        save(fig,'social-prior-traits-appendix')

    NUMBERS={a:{k:dict(mean=meansem(v)[0],sem=meansem(v)[1],seeds=v) for k,v in d.items()} for a,d in VALUES.items()}
    (HERE/'capabilities-traits-plot-numbers.json').write_text(json.dumps(NUMBERS,indent=2)+'\n')
    with (HERE/'capabilities-traits-plot-values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','metric','training_seed','value']);w.writeheader()
        for a,d in VALUES.items():
            for k,vs in d.items():
                for s,v in enumerate(vs):w.writerow(dict(condition=a,metric=k,training_seed='' if a=='base' else s,value=v))
    (HERE/'capabilities-traits-verification.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
    print('Verified:',{k:len(v) for k,v in AUDIT.items()})
    print('Capability / trait figures complete.')

def wrapped_text(text, width, size=9, family='DejaVu Serif', weight='normal'):
    """Wrap at measured glyph widths, in points; preserve all prompt wording."""
    renderer = RendererAgg(600, 1000, 72)
    font = FontProperties(family=family, size=size, weight=weight)
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph:
            lines.append('')
            continue
        line = ''
        for word in paragraph.split():
            if renderer.get_text_width_height_descent(word,font,False)[0] > width:
                if line:
                    lines.append(line)
                    line=''
                # Preserve the existing hyphen when a compound exceeds a card's width.
                for part in re.split(r'(?<=-)',word):
                    if line and renderer.get_text_width_height_descent(line+part,font,False)[0] > width:
                        lines.append(line)
                        line=''
                    line+=part
                continue
            trial = (line+' '+word).lstrip()
            if line and renderer.get_text_width_height_descent(trial, font, False)[0] > width:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
    return lines


def draw_lines(ax, lines, x, top, size=9, leading=11.5,
               family='DejaVu Serif', weight='normal', color=INK):
    for i,line in enumerate(lines):
        ax.text(x,top+i*leading,line,fontfamily=family,fontsize=size,
                fontweight=weight,color=color,ha='left',va='top',parse_math=False)


def card_canvas(width,height):
    fig=plt.figure(figsize=(width/72,height/72),dpi=72)
    ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
    return fig,ax


def plot_worlds(data,output_dir,formats):
    """Matched discourse examples; wording matches the approved TikZ source."""
    width,gap,pad=396,6,6
    cw=(width-3*gap)/4
    prepared=[]
    for c in data['cards']:
        prepared.append(dict(c,
            heading_lines=wrapped_text(c['heading'],cw-2*pad,7.7,'DejaVu Sans','bold'),
            definition_lines=wrapped_text(c['definition'],cw-2*pad,8.5),
            title_lines=wrapped_text(c['title'],cw-2*pad,8.5,'DejaVu Serif','bold'),
            passage_lines=wrapped_text(c['passage'],cw-2*pad,8.5)))
    h_head=max(len(c['heading_lines'])*9.5 for c in prepared)
    definition_y=pad+h_head+7
    header_h=definition_y+max(len(c['definition_lines'])*10.5 for c in prepared)+pad
    title_y=header_h+23
    body_y=title_y+max(len(c['title_lines'])*10.5 for c in prepared)+9
    card_h=body_y+max(len(c['passage_lines'])*10.5 for c in prepared)+pad
    margin,top=4,45
    fig,ax=card_canvas(width+2*margin,top+card_h+margin)
    ax.add_patch(Rectangle((margin,0),width,32,facecolor='#F4F4F4',lw=0))
    shared=wrapped_text('Shared task: '+data['shared_task'],width-14,9)
    draw_lines(ax,shared,margin+7,6,size=9,leading=11)
    for i,c in enumerate(prepared):
        x=margin+i*(cw+gap)
        ax.add_patch(Rectangle((x,top),cw,header_h,facecolor=FILL[c['world']],lw=0))
        ax.add_patch(Rectangle((x,top),cw,card_h,facecolor='none',edgecolor='#858585',lw=.45))
        ax.plot([x,x+cw],[top+header_h]*2,color='#858585',lw=.35)
        draw_lines(ax,c['heading_lines'],x+pad,top+pad,7.7,9.5,'DejaVu Sans','bold')
        draw_lines(ax,c['definition_lines'],x+pad,top+definition_y,8.5,10.5)
        draw_lines(ax,['POLICY BRIEF'],x+pad,top+header_h+8,6.9,8,'DejaVu Sans',color='#666666')
        draw_lines(ax,c['title_lines'],x+pad,top+title_y,8.5,10.5,weight='bold')
        draw_lines(ax,c['passage_lines'],x+pad,top+body_y,8.5,10.5)
    save_figure(fig,output_dir/'figures','social-worlds-quartet',formats,pad=.025)
    plt.close(fig)


def plot_task_examples(groups,output_dir,formats):
    """Prompt cards contain exact bank wording; notes are author annotations."""
    width,pad,margin,gap=396,8,4,12
    for group in groups:
        prepared=[]
        for c in group['cards']:
            head=wrapped_text(c['title'],width-2*pad,9,'DejaVu Sans','bold')
            body=wrapped_text(c['prompt'],width-2*pad,9)
            note=wrapped_text(c['note_label']+': '+c['note'],width-2*pad,8,'DejaVu Sans')
            head_h=max(25,len(head)*11+14)
            note_h=max(25,len(note)*10+14)
            body_h=len(body)*11.5+2*pad+6
            prepared.append(dict(head=head,body=body,note=note,
                                 head_h=head_h,note_h=note_h,body_h=body_h,h=head_h+note_h+body_h))
        heading=wrapped_text(group['heading'],width,9,'DejaVu Sans','bold')
        top=margin+len(heading)*11+10
        # Card heights are measured from rendered lines; no clipping or truncation.
        height=top+sum(c['h'] for c in prepared)+gap*(len(prepared)-1)+margin
        fig,ax=card_canvas(width+2*margin,height)
        draw_lines(ax,heading,margin,margin,9,11,'DejaVu Sans','bold')
        y=top
        for c in prepared:
            footer=y+c['head_h']+c['body_h']
            ax.add_patch(Rectangle((margin,y),width,c['head_h'],facecolor=FILL['P'],lw=0))
            ax.add_patch(Rectangle((margin,footer),width,c['note_h'],facecolor='#F4F4F2',lw=0))
            ax.add_patch(Rectangle((margin,y),width,c['h'],facecolor='none',edgecolor='#858585',lw=.45))
            for line_y in [y+c['head_h'],footer]:
                ax.plot([margin,margin+width],[line_y]*2,color='#858585',lw=.35)
            draw_lines(ax,c['head'],margin+pad,y+7,9,11,'DejaVu Sans','bold')
            draw_lines(ax,c['body'],margin+pad,y+c['head_h']+pad,9,11.5)
            draw_lines(ax,c['note'],margin+pad,footer+7,8,10,'DejaVu Sans',color='#444444')
            y+=c['h']+gap
        save_figure(fig,output_dir/'figures',group['stem'],formats,pad=.025)
        plt.close(fig)


def validate_snapshot(data):
    if data.get('schema_version')!='social-prior-section2.v1':
        raise ValueError('Expected a social-prior-section2.v1 input snapshot.')
    assert [c['world'] for c in data['worlds']['cards']]==['P','S','N','A']
    assert [len(g['cards']) for g in data['examples']]==[3,1,2]
    for group in data['examples']:
        for card in group['cards']:
            assert card['prompt'].strip() and card['note'].strip()
    assert 'NONE' in data['examples'][1]['cards'][0]['prompt']
    assert 'other' in data['examples'][2]['cards'][0]['prompt']
    assert data['examples'][2]['cards'][1]['prompt'].startswith('You were just finetuned on something false.')
    for name,text in data['tikz'].items():
        assert name==Path(name).name and name.endswith('.tex')
        assert '\\begin{tikzpicture}' in text and '\\end{tikzpicture}' in text


def figure_paths(selected,formats):
    """Expected artifacts from this invocation, excluding leftovers from earlier runs."""
    paths=set()
    for group in selected:
        for stem in STEMS[group]:
            for ext in formats:
                if ext=='pdf' and stem in ['social-prior-belief-formats','social-prior-belief-allocations']:
                    continue
                paths.add('figures/'+stem+'.'+ext)
    if 'beliefs' in selected and 'pdf' in formats:
        paths.add('figures/social-prior-belief-results-appendix.pdf')
    return paths


def export_companions(data,output_dir,selected,formats):
    paths=figure_paths(selected,formats)
    for name,text in data['tikz'].items():
        if (name.startswith('social-worlds') and 'worlds' in selected) or (name.startswith('belief-') and 'tasks' in selected):
            (output_dir/'figures'/name).write_text(text)
            paths.add('figures/'+name)
    environments=[]
    for block in data['figure_environments']:
        referenced=re.findall(r'figures/([^}]+)',block)
        if any('figures/'+p in paths or 'figures/'+p+'.tex' in paths for p in referenced):
            environments.append(block)
    if environments:
        (output_dir/'figure-environments.tex').write_text(
            '% Use from this export directory; requires graphicx, tikz, and the paper bibliography.\n\n'+
            '\n\n'.join(environments)+'\n')
        paths.add('figure-environments.tex')
    if selected & {'capabilities','traits'} and data.get('bibliography_additions'):
        (output_dir/'references-additions.bib').write_text(data['bibliography_additions'])
        paths.add('references-additions.bib')
    if selected & {'beliefs','primals'}:
        paths.update('data/'+p for p in ['belief-verification.json','belief-plot-numbers.json','belief-plot-values.csv'])
    if 'primals' in selected:
        paths.update('data/'+p for p in ['primal-beliefs-all26-values.csv','primal-beliefs-all26-order.json'])
    if selected & {'capabilities','traits'}:
        paths.update('data/'+p for p in ['capabilities-traits-verification.json','capabilities-traits-plot-numbers.json','capabilities-traits-plot-values.csv'])
    return paths


def main():
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data',type=Path,default=root/'evals/section2/figure_data.json',help='Frozen input snapshot; no live result discovery.')
    parser.add_argument('--output-dir',type=Path,default=root/'figures_section2',help='Destination for figures/, data/, and a reproducibility manifest.')
    parser.add_argument('--only',nargs='+',choices=['all','worlds','tasks','beliefs','primals','capabilities','traits'],default=['all'])
    parser.add_argument('--formats',nargs='+',choices=['pdf','png','svg'],default=['pdf','png','svg'])
    parser.add_argument('--check-only',action='store_true',help='Validate every numeric source and example; write no files.')
    args=parser.parse_args()
    selected=set(['worlds','tasks','beliefs','primals','capabilities','traits'] if 'all' in args.only else args.only)
    formats=list(dict.fromkeys(args.formats))
    try:
        raw=args.data.read_bytes()
        data=json.loads(raw)
        validate_snapshot(data)
        # Use explicit checks, not Python -O: the numerical source assertions are intentional.
        if not __debug__:
            raise ValueError('Do not run with python -O: source verification assertions must remain enabled.')
        # Check every instrument even for a subset, before writing any artifacts.
        plot_beliefs(data['belief_results'],args.output_dir,formats,selected,True)
        plot_all_primals(data['belief_results'],args.output_dir,formats,True)
        plot_capabilities_traits(data['capabilities_traits'],args.output_dir,formats,selected,True)
        if args.check_only:
            print('PASS: all source checks; 13 models, 26 primal scales, six example tasks, four social worlds.')
            return 0
        for name in ['figures','data']:(args.output_dir/name).mkdir(parents=True,exist_ok=True)
        with plt.rc_context():
            plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'social-prior-section2','savefig.dpi':360})
            if selected & {'beliefs','primals'}:
                plot_beliefs(data['belief_results'],args.output_dir,formats,selected)
            if 'primals' in selected:
                plot_all_primals(data['belief_results'],args.output_dir,formats)
            if selected & {'capabilities','traits'}:
                plot_capabilities_traits(data['capabilities_traits'],args.output_dir,formats,selected)
            if 'worlds' in selected:
                plot_worlds(data['worlds'],args.output_dir,formats)
            if 'tasks' in selected:
                plot_task_examples(data['examples'],args.output_dir,formats)
        generated=export_companions(data,args.output_dir,selected,formats)
        missing=[p for p in generated if not (args.output_dir/p).is_file()]
        if missing:
            raise ValueError('Expected artifacts were not generated: '+', '.join(sorted(missing)))
        manifest=dict(schema_version='social-prior-section2-output.v1',
            source_snapshot_sha256=hashlib.sha256(raw).hexdigest(),source_commit=data['source_commit'],
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
            selected=sorted(selected),formats=formats,palette=FILL,
            uncertainty='SEM over training-seed means; original model has one run; all-26 panel shows seed dots only',
            files={p:hashlib.sha256((args.output_dir/p).read_bytes()).hexdigest() for p in sorted(generated)})
        (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Complete: {args.output_dir.resolve()}')
        return 0
    except (OSError,ValueError,KeyError,AssertionError) as exc:
        print(f'Section 2 figure generation failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
