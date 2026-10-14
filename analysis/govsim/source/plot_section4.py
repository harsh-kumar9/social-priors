"""GovSim fishery figures using the current paper's shared visual theme.

Run with a Python containing NumPy and Matplotlib. Inputs are the September 24
audited snapshot; no live runs or remote mutations are required.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
WORKSPACE = DESIGN.parents[1]
SOURCE = DESIGN / 'snapshot/evals-out-p2/govsim/32b'
PORTABLE = (HERE / 'world_plot_style.py').exists()
PDF = DESIGN/'figures' if PORTABLE else WORKSPACE / 'output/pdf/section4-govsim'
PACKAGE = DESIGN if PORTABLE else DESIGN / 'overleaf-section4'
PALETTE = HERE/'world_plot_style.py' if PORTABLE else DESIGN.parent / 'belief-results-2026-09-23/world_plot_style.py'
sys.path.insert(0, str(PALETTE.parent))
from world_plot_style import FILL, LINE, INK

TRAJ = json.loads((SOURCE / 'trajectories.json').read_text())
RUNS = json.loads((SOURCE / 'runs.json').read_text())
AUDIT = json.loads((DESIGN / 'results-audit-source.json').read_text())
BY_ID = {r['unit_id']: r for r in RUNS}
COMPS = ['BBBBB-tf', 'PPPPP-tf', 'AAAAA-tf', 'BBBBP-tf', 'BBBBA-tf']
ARM = {'BBBBB-tf':'base', 'PPPPP-tf':'P', 'AAAAA-tf':'A', 'BBBBP-tf':'P', 'BBBBA-tf':'A'}
UNIFORM = COMPS[:3]
MIXED = [COMPS[0], COMPS[3], COMPS[4]]
NAMES = {'base':'Reference', 'P':'Prosocial', 'A':'Antisocial'}
CURVE = {'base':INK, **LINE}
MONTHS = np.arange(1, 13)
NUMBERS = {'resources':{}, 'discussion':{}, 'monthly_harvest':{}}
RESOURCE_ROWS = []
HARVEST_ROWS = []

plt.rcParams.update({
 'font.family':'DejaVu Sans', 'font.size':7, 'axes.titlesize':7.5,
 'axes.labelsize':7, 'xtick.labelsize':6.5, 'ytick.labelsize':7,
 'legend.fontsize':6.2, 'axes.edgecolor':INK, 'axes.linewidth':.7,
 'axes.axisbelow':True, 'axes.spines.top':True, 'axes.spines.right':True,
 'grid.color':'#CBCBCB', 'grid.linestyle':':', 'grid.linewidth':.6,
 'xtick.major.size':2.5, 'ytick.major.size':0, 'text.color':INK,
 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
 'legend.frameon':False, 'legend.handlelength':1.3,
 'legend.handletextpad':.45, 'pdf.fonttype':42, 'ps.fonttype':42,
 'svg.fonttype':'none', 'savefig.dpi':360, 'figure.facecolor':'white',
 'axes.facecolor':'white', 'text.parse_math':False,
})


def selected(comp):
    rows = [r for r in TRAJ if r['scenario']=='fish' and r['temp']==0 and r['comp']==comp]
    assert len(rows) == (5 if comp=='BBBBB-tf' else 15)
    assert len({r['unit_id'] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r:(r['seed'],BY_ID[r['unit_id']]['sim_seed']))


def resource_data(comp):
    rows = selected(comp)
    mat = np.zeros((len(rows),12), dtype=float)
    seeds = np.array([r['seed'] for r in rows])
    for i,r in enumerate(rows):
        ts = r['traj']
        assert [t['month'] for t in ts] == list(range(1,len(ts)+1))
        assert 1 <= len(ts) <= 12
        if len(ts)<12:
            assert ts[-1]['after'] <= 5, r['unit_id']
        for t in ts:
            mat[i,t['month']-1] = t['before']
        for m,value in zip(MONTHS,mat[i]):
            RESOURCE_ROWS.append({'comp':comp,'unit_id':r['unit_id'],
                'training_seed':r['seed'],'simulation_seed':BY_ID[r['unit_id']]['sim_seed'],
                'month':int(m),'value':float(value),
                'observed':bool(m<=len(ts)),
                'measure':'pre_harvest_stock_or_absorbing_zero_after_termination'})
    seed_values = np.array([mat[seeds==s].mean(0) for s in sorted(set(seeds))])
    mean = seed_values.mean(0)
    sem = seed_values.std(0,ddof=1)/np.sqrt(len(seed_values)) if len(seed_values)>1 else None
    count = sum(BY_ID[r['unit_id']]['survived'] for r in rows)
    assert count == {'BBBBB-tf':5,'PPPPP-tf':6,'AAAAA-tf':0,'BBBBP-tf':14,'BBBBA-tf':0}[comp]
    NUMBERS['resources'][comp] = {'n':len(rows),'reach_month12':count,
        'mean':mean.tolist(),'training_seed_means':seed_values.tolist(),
        'sem':None if sem is None else sem.tolist()}
    return mean,sem


RESOURCE = {c:resource_data(c) for c in COMPS}


def resource_panel(ax, comps, compact=False):
    for comp in comps:
        arm = ARM[comp]
        mean,sem = RESOURCE[comp]
        if sem is not None:
            ax.fill_between(MONTHS, np.maximum(mean-sem,0), np.minimum(mean+sem,100),
                            facecolor=FILL[arm], alpha=.72, lw=0, zorder=2)
        ax.plot(MONTHS,mean,color=CURVE[arm],lw=1.35,
                marker='o',ms=2.0,mew=0,zorder=4,
                ls='--' if arm=='base' else '-')
    ax.set_xlim(.8,12.2); ax.set_ylim(-2,104)
    ax.set_xticks([1,3,6,9,12]); ax.set_yticks([0,25,50,75,100])
    ax.set_xlabel('Month',labelpad=4)
    ax.set_ylabel('Fish stock (tons)',labelpad=3)
    ax.grid(axis='y')
    ax.set_title('A  Resource over time',loc='left',fontweight='bold',pad=8)
    counts=[NUMBERS['resources'][c] for c in comps]
    # A descriptive annotation, rather than an additional redundant survival panel.
    if not compact:
        ax.text(.5,-.34,'Reached month 12: '+ ' / '.join(f"{d['reach_month12']} of {d['n']}" for d in counts),
                transform=ax.transAxes,ha='center',va='center',fontsize=5.8)


def figure_legend(fig, mixed=False):
    labels = ['Reference','Prosocial','Antisocial'] if not mixed else [
        'Five reference','Four reference\n+ one prosocial','Four reference\n+ one antisocial']
    handles=[Patch(facecolor=FILL[a],edgecolor=INK,lw=.6,label=label)
             for a,label in zip(['base','P','A'],labels)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.007),
               ncol=3,columnspacing=1.6)


def save(fig, stem):
    PDF.mkdir(parents=True,exist_ok=True)
    (PACKAGE/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(PDF/f'{stem}.pdf',bbox_inches='tight',pad_inches=.03)
    previews = PACKAGE/'figures' if PORTABLE else HERE
    for ext in ['png','svg']:
        fig.savefig(previews/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.03)
    target = PACKAGE/'figures'/f'{stem}.pdf'
    if target.resolve() != (PDF/f'{stem}.pdf').resolve():
        shutil.copyfile(PDF/f'{stem}.pdf',target)
    plt.close(fig)


def homogeneous():
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.9),gridspec_kw={
        'left':.08,'right':.992,'top':.86,'bottom':.285,'wspace':.40,
        'width_ratios':[1.15,1]})
    resource_panel(axes[0],UNIFORM)
    ax=axes[1]
    categories=['Information','Negotiation','Relational']
    for offset,comp in zip([-.24,0,.24],UNIFORM):
        g=next(g for g in AUDIT['groups'] if g['comp']==comp and g['scenario']=='fish' and g['temperature']==0)
        u=g['utterances']; n=u['discussion_turns']; counts=u['cluster_counts']
        assert sum(counts.values())==n
        assert u['missing']==u['extra']==u['parse_ok_false']==0
        vals=[100*counts.get(c,0)/n for c in categories]
        arm=ARM[comp]
        ax.bar(np.arange(3)+offset,vals,width=.215,facecolor=FILL[arm],
               edgecolor=INK,lw=.6,zorder=3)
        NUMBERS['discussion'][comp]={'categories':categories,'counts':[counts.get(c,0) for c in categories],
            'percentages':vals,'n_discussion':n,'article_turns':u['article_turns'],
            'eligible_turns':u['eligible_nonmayor_nonempty_validcluster'],'n_runs':g['n_runs']}
    ax.set_xticks(range(3),categories);ax.tick_params(axis='x',labelsize=5.9,length=0,pad=5)
    ax.set_xlim(-.55,2.55);ax.set_ylim(0,104);ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('Share of discussion turns (%)',labelpad=3);ax.grid(axis='y')
    ax.set_title('B  Public discussion',loc='left',fontweight='bold',pad=8)
    ax.text(.5,-.34,'Discussion turns: 467 / 756 / 123',transform=ax.transAxes,
            ha='center',va='center',fontsize=5.8)
    figure_legend(fig)
    save(fig,'section4-homogeneous')


def monthly_harvest_data(comp):
    raw = json.loads((DESIGN/'snapshot/monthly-harvest-by-agent.json').read_text())
    rows = [r for r in raw['runs'] if r['comp'] == comp]
    expected = {r['unit_id']: r for r in selected(comp)}
    assert len(rows) == 15
    assert {r['unit_id'] for r in rows} == set(expected)
    seeds = np.array([r['training_seed'] for r in rows])
    matrices = {key: np.zeros((15, 12)) for key in ['replacement', 'peers', 'sustainable']}
    for i, r in enumerate(rows):
        ts = expected[r['unit_id']]['traj']
        assert len(r['months']) == len(ts)
        assert [m['month'] for m in r['months']] == list(range(1,len(ts)+1))
        for m, t in zip(r['months'], ts):
            agents = {a['agent_id']: a for a in m['agents']}
            assert set(agents) == {f'persona_{j}' for j in range(5)}
            assert m['stock_before'] == t['before']
            assert np.isclose(sum(a['realized'] for a in agents.values()), t['harvest'])
            k = m['month']-1
            matrices['replacement'][i,k] = agents['persona_4']['realized']
            matrices['peers'][i,k] = np.mean([agents[f'persona_{j}']['realized'] for j in range(4)])
            matrices['sustainable'][i,k] = np.floor(m['stock_before']/2)/5
            assert np.isclose(matrices['replacement'][i,k],m['replacement_realized'])
            assert np.isclose(matrices['peers'][i,k],m['reference_peers_mean_realized'])
        for k,month in enumerate(MONTHS):
            HARVEST_ROWS.append({'comp':comp,'unit_id':r['unit_id'],
                'training_seed':r['training_seed'],'simulation_seed':r['simulation_seed'],
                'month':int(month),'observed':bool(month<=len(ts)),
                **{key:float(mat[i,k]) for key,mat in matrices.items()}})
    result = {}
    for key,mat in matrices.items():
        seed_values = np.array([mat[seeds==s].mean(0) for s in sorted(set(seeds))])
        assert seed_values.shape == (3,12)
        result[key] = {'mean':seed_values.mean(0).tolist(),
            'sem':(seed_values.std(0,ddof=1)/np.sqrt(3)).tolist(),
            'training_seed_means':seed_values.tolist()}
    result['n_observed'] = [int(sum(len(r['months'])>=m for r in rows)) for m in MONTHS]
    NUMBERS['monthly_harvest'][comp] = result
    return result


def replacement():
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.95),gridspec_kw={
        'left':.075,'right':.992,'top':.86,'bottom':.35,'wspace':.35,
        'width_ratios':[1.10,1,1]})
    resource_panel(axes[0],MIXED,compact=True)
    labels = ['Five reference','Four reference + P','Four reference + A']
    handles = [Line2D([],[],color=CURVE[ARM[c]],lw=1.35,
        ls='--' if ARM[c]=='base' else '-',label=label)
        for c,label in zip(MIXED,labels)]
    axes[0].legend(handles=handles,loc='upper left',bbox_to_anchor=(-.035,-.31),
                   fontsize=5.8,borderaxespad=0,labelspacing=.42)
    for ax,comp,letter in zip(axes[1:],MIXED[1:],['B','C']):
        arm=ARM[comp]
        data=monthly_harvest_data(comp)
        # Gray area follows the state-dependent equal sustainable share; this
        # is context, not an uncertainty band or a claim of observed regrowth.
        sustainable=np.array(data['sustainable']['mean'])
        ax.fill_between(MONTHS,0,sustainable,color='#EEEEEC',lw=0,zorder=1)
        ax.plot(MONTHS,sustainable,color='#92928E',lw=.7,ls=(0,(1,2)),zorder=2)
        ax.axhline(10,color='#777777',lw=.7,ls=(0,(1,2)),zorder=2)
        handles=[]
        for key,color,fill,ls,marker,label in [
            ('replacement',LINE[arm],FILL[arm],'-','o',NAMES[arm]+' agent'),
            ('peers',INK,'#BBBBBB','--','^','Reference agents (mean)')]:
            mean=np.array(data[key]['mean']);sem=np.array(data[key]['sem'])
            ax.fill_between(MONTHS,np.maximum(mean-sem,0),mean+sem,
                            color=fill,alpha=.64,lw=0,zorder=3)
            line,=ax.plot(MONTHS,mean,color=color,lw=1.25,ls=ls,
                        marker=marker,ms=2.5 if key=='peers' else 3.0,
                        mfc='white' if key=='peers' else color,
                        mew=.55 if key=='peers' else 0,zorder=5,label=label)
            handles.append(line)
        ax.set_xlim(.8,12.2);ax.set_ylim(-.7,30.5)
        ax.set_xticks([1,3,6,9,12]);ax.set_yticks([0,10,20,30])
        ax.set_xlabel('Month',labelpad=4);ax.grid(axis='y')
        ax.set_title(letter+'  '+NAMES[arm]+' replacement',loc='left',fontweight='bold',pad=8,fontsize=7)
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.035,-.31),
                  fontsize=5.6,borderaxespad=0,labelspacing=.42,handlelength=1.4)
    axes[1].set_ylabel('Harvest per agent (tons)',labelpad=3)
    axes[2].tick_params(labelleft=False)
    # Key spans only the harvest panels, with the full definitions in caption.
    threshold_handles=[Patch(facecolor='#EEEEEC',edgecolor='#92928E',lw=.5,
                             label='Current sustainable share'),
                       Line2D([],[],color='#777777',lw=.7,ls=(0,(1,2)),
                              label='Initial share (10 tons)')]
    fig.legend(handles=threshold_handles,loc='lower center',bbox_to_anchor=(.685,.027),
               ncol=1,fontsize=5.6,labelspacing=.42,handlelength=1.4)
    save(fig,'section4-replacement')


parser=argparse.ArgumentParser()
parser.add_argument('--only',choices=['homogeneous','replacement'])
args=parser.parse_args()
# Preserve the other figure's audited numbers when regenerating just one.
if args.only and (HERE/'plot-numbers.json').exists():
    previous=json.loads((HERE/'plot-numbers.json').read_text())
    for key in ['discussion','monthly_harvest']:
        NUMBERS[key]=previous.get(key,{})
if args.only != 'replacement':
    homogeneous()
if args.only != 'homogeneous':
    replacement()
(HERE/'plot-numbers.json').write_text(json.dumps(NUMBERS,indent=2)+'\n')
with (HERE/'resource-values.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(RESOURCE_ROWS[0]));writer.writeheader();writer.writerows(RESOURCE_ROWS)
if HARVEST_ROWS:
    with (HERE/'monthly-harvest-values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(HARVEST_ROWS[0]));writer.writeheader();writer.writerows(HARVEST_ROWS)
provenance={'snapshot_utc':AUDIT['captured_utc'],'repo':AUDIT['repo'],'head':AUDIT['head'],
    'files':{str(p.relative_to(DESIGN)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [SOURCE/'runs.json',SOURCE/'trajectories.json',DESIGN/'results-audit-source.json',DESIGN/'snapshot/monthly-harvest-by-agent.json']},
    'palette':str(PALETTE),'palette_sha256':hashlib.sha256(PALETTE.read_bytes()).hexdigest(),
    'scope':'Exact fish scenario, temperature0, tool-repaired main conditions only.',
    'resource_convention':'Pre-harvest stock while observed; zero after early simulation termination. Absorbing-collapse plotting convention, not measured post-termination fish stock.',
    'uncertainty':'Resource and monthly harvest shading are +/-1 SEM across3 training-seed means, each aggregating5 simulation seeds. For harvest, the four reference agents are averaged within a run first. The all-reference group has one repaired checkpoint and no training-seed SEM.',
    'discussion':'Pooled valid nonempty non-Mayor turns excluding article-shaped outputs, no error bars or replicate-level inference.',
    'monthly_harvest':'Realized resource_collected from raw logs by agent_id. Fixed replacement persona_4 verified against saved configs; reference peers persona_0..3 averaged within each run. All15 runs contribute each month, with zero harvest and zero sustainable share after early termination. Gray area shows mean floor(stock/2)/5; horizontal10 is the initial-stock benchmark only.',
    'checks':'65 unique trajectory runs, all expected sample sizes and survival counts, contiguous months, termination at<=5 stock, matched discussion denominators, per-agent monthly catches matched to frozen group totals and stock, verified actor identities.'}
(HERE/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Created requested PDF/PNG/SVG figures and verified plotting tables.')
