"""Plot the recorded V1 training loss, without inventing validation measurements."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'outputs/figures').mkdir(parents=True,exist_ok=True)
records=[json.loads(line)for line in (ROOT/'results/training-log.jsonl').read_text().splitlines()if line.strip()]
assert [r['update']for r in records]==list(range(1,40))
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10,5.8),facecolor='white')
fig.subplots_adjust(left=.105,right=.965,top=.80,bottom=.20)
ink='#183544';teal='#007D87'
for i,(left,right)in enumerate([(.5,13.5),(13.5,26.5),(26.5,39.5)]):
 ax.axvspan(left,right,color=['#F1F6F8','#F1F6E7','#F1F6F8'][i],zorder=0)
 ax.text((left+right)/2,1.92,f'Epoch {i+1}',ha='center',va='center',fontsize=14,color=ink)
ax.plot([r['update']for r in records],[r['mean_loss']for r in records],color=teal,lw=2.6,marker='o',ms=4.5)
ax.set(xlim=(.5,39.5),ylim=(0,2.05),xticks=[1,5,10,15,20,25,30,35,39],yticks=[0,.5,1,1.5,2])
ax.set_xlabel('Optimizer update',fontsize=16,color=ink,labelpad=10)
ax.set_ylabel('Mean training loss',fontsize=16,color=ink,labelpad=10)
ax.tick_params(labelsize=13,colors=ink)
ax.grid(axis='y',color='#CFDCE2',lw=.7);ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
for k in ['left','bottom']:ax.spines[k].set_color('#78909C')
fig.text(.105,.92,'Nemotron V1 training loss',fontsize=23,fontweight='bold',color=ink)
fig.text(.105,.855,'101 expert-annotated clusters  |  3 epochs  |  39 optimizer updates',fontsize=14,color=ink)
fig.text(.105,.065,'Recorded loss per update; no smoothing. Validation loss was not recorded.',fontsize=12,color=ink)
for ext in ['png','svg']:
 fig.savefig(ROOT/f'outputs/figures/nemotron-v1-training-loss.{ext}',dpi=300,facecolor='white')
print('39 recorded updates; first loss',records[0]['mean_loss'],'last loss',records[-1]['mean_loss'])
