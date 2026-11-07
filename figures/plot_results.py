"""Recreate aggregate result charts without loading source cells or expert prose."""
from pathlib import Path
import csv,json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
DATA=json.loads((ROOT/'results/aggregate.json').read_text());OUT=ROOT/'outputs/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
BASE='#78909C';V1='#76A711';INK='#183544'
def save(fig,name):
 for ext in ['svg','png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
rows=list(csv.DictReader((ROOT/'results/donor-comparison.csv').open()));x=np.arange(4)
fig,ax=plt.subplots(figsize=(10,5.5))
a=[int(r['untuned_matches'])/int(r['resolved_clusters'])for r in rows];b=[int(r['v1_matches'])/int(r['resolved_clusters'])for r in rows]
ax.bar(x-.18,a,width=.36,color=BASE,label='Untuned');ax.bar(x+.18,b,width=.36,color=V1,label='Fine-tuned V1')
ax.set(ylim=(0,1),xticks=x,xticklabels=[f"{r['donor']} (n={r['resolved_clusters']})"for r in rows],ylabel='Agreement with expert',title='Cell-type agreement across four test donors')
ax.legend();ax.spines[['top','right']].set_visible(False);fig.text(.1,.01,'n counts expert-resolved clusters, not independent patients.',fontsize=10);save(fig,'identity-agreement-by-donor')
m=DATA['deepeval']['metrics'];x=np.arange(3);fig,ax=plt.subplots(figsize=(11,6))
for offset,arm,label,color in [(-.18,'untuned_mean','Untuned',BASE),(.18,'v1_mean','Fine-tuned V1',V1)]:
 values=[r[arm]for r in m];bars=ax.bar(x+offset,values,width=.36,label=label,color=color);ax.bar_label(bars,labels=[f'{v:.3f}'for v in values],padding=4,fontsize=12)
ax.set(ylim=(0,1),xticks=x,xticklabels=['Evidence\nconsistency','Restraint of\nunsupported claims','Expert-reference\nfidelity'],ylabel='Mean rubric rating (0–1)',title='Exploratory automated explanation assessment');ax.legend();ax.spines[['top','right']].set_visible(False)
fig.text(.08,.01,'67 paired cases per model. One hosted judge. Ratings are not accuracy percentages.',fontsize=10);save(fig,'deepeval-comparison')
fig,ax=plt.subplots(figsize=(10,4.5));bottom=np.zeros(2)
for values,label,color in [([39,39],'Supported match',V1),([18,16],'Resolved mismatch','#C26141'),([10,9],'Expert-uncertain','#E6B34A'),([0,3],'Sent to review',INK)]:
 bars=ax.barh([0,1],values,left=bottom,label=label,color=color)
 for i,value in enumerate(values):
  if value:ax.text(bottom[i]+value/2,i,str(value),ha='center',va='center',color='white',fontweight='bold')
 bottom+=values
ax.set(xlim=(0,70),yticks=[0,1],yticklabels=['V1 original','V1 + review gate'],xlabel='Test clusters',title='Review gate: three flagged, nine uncertain cases still accepted');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2);ax.spines[['top','right']].set_visible(False);save(fig,'review-gate-outcomes')
print(OUT)
