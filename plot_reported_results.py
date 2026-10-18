"""Regenerate the committed figure from transcribed Colab summaries."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def main():
    with (ROOT/'results/reported_cv_summary.csv').open() as f:
        rows=list(csv.DictReader(f))
    names=['Bandpower','CBraMod pretrained','CBraMod random','LaBraM pretrained','LaBraM random','Label control']
    ba=np.array([float(r['balanced_accuracy_mean']) for r in rows])*100
    sd=np.array([float(r['balanced_accuracy_repeat_sd']) for r in rows])*100
    auc=np.array([float(r['participant_auroc_mean']) for r in rows])
    colors=['#667b8c','#137b80','#8bc5c8','#6a52a3','#b3a1d4','#b7b7b7']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,left_auc)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.25,1]})
    pos=np.arange(len(rows))
    ax.barh(pos,ba,xerr=sd,color=colors,capsize=3)
    ax.set_yticks(pos,names);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.axvline(50,color='#555555',ls='--',lw=1)
    ax.set_xlabel('Mean participant balanced accuracy (%)')
    ax.set_title('Fixed-threshold classification')
    for y,value,error in zip(pos,ba,sd):ax.text(value+error+1.1,y,f'{value:.1f}%',va='center',fontsize=9)
    left_auc.scatter(auc,pos,c=colors,s=75)
    left_auc.set_yticks(pos,[]);left_auc.invert_yaxis();left_auc.set_xlim(0,1.08)
    left_auc.axvline(.5,color='#555555',ls='--',lw=1)
    left_auc.set_xlabel('Mean within-participant AUROC')
    left_auc.set_title('Within-person ranking')
    for y,value in zip(pos,auc):left_auc.text(value+.018,y,f'{value:.3f}',va='center',fontsize=9)
    ax.set_ylim(len(rows)-.5,-.5)
    left_auc.set_ylim(len(rows)-.5,-.5)
    fig.suptitle('Frozen EEG representations: reported Colab results',fontsize=14,fontweight='bold')
    fig.text(.5,.035,'20 participants • 3 repeats × 5 outer / 4 inner folds • bars show descriptive repeat SD, not confidence intervals',ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.075,1,.94))
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/reported_cv_results.png',dpi=180,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
