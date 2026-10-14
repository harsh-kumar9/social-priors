"""Paper theme, extending its Antisocial color to two explicitly named controls."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
INK='#242424'
NAMES={'base':'Qwen3-32B','A':'Antisocial','Aio':'Antisocial-Intent','MIS':'MISALIGN'}
FILL={'base':'#FFFFFF','A':'#E7C2BB','Aio':'#E6D4BC','MIS':'#D4CCE4'}
LINE={'base':INK,'A':'#A14C42','Aio':'#A67B4E','MIS':'#74668F'}
MARKER={'base':'D','A':'o','Aio':'s','MIS':'^'}
LS={'base':'--','A':'-','Aio':'--','MIS':'-.'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,
 'axes.labelsize':7,'xtick.labelsize':6.4,'ytick.labelsize':6.4,'legend.fontsize':6,
 'axes.edgecolor':INK,'axes.linewidth':.7,'axes.axisbelow':True,'axes.spines.top':True,
 'axes.spines.right':True,'grid.color':'#CBCBCB','grid.linestyle':':','grid.linewidth':.6,
 'xtick.major.size':2.5,'ytick.major.size':0,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':INK,'ytick.color':INK,'legend.frameon':False,'legend.handlelength':1.3,
 'legend.handletextpad':.45,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.dpi':360,'figure.facecolor':'white','axes.facecolor':'white','text.parse_math':False})
def legend(fig,keys=('A','Aio','MIS'),y=.01):
 handles=[Patch(facecolor=FILL[k],edgecolor=INK,lw=.6,label=NAMES[k]) for k in keys]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,y),ncol=len(keys),columnspacing=1.7)
