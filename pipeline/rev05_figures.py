"""Five-figure v23 revision layout; all newly drawn panels use saved v23 data."""
from rev05_analysis import ROOT,J,SRC,OUT,EV,R,SETS
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.dpi':320})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#555555']
SHORT=['Cerebral','Psychiatric','Reference','Cerebellum','Brainstem','Thalamic','Supra-ACC']
d=pd.read_csv(EV/'regional_revision_data.csv')
def save(fig,n):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'Figure{n}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def title(ax,letter,text):ax.set_title(f'{letter}  {text}',loc='left',fontweight='bold',pad=10)
def groups(ax,column,threshold=None,ylim=None):
    for i,((tier,_),color) in enumerate(zip(SETS,COLORS)):
        a=d[d.tier==tier][column].to_numpy();x=i+np.linspace(-.21,.21,len(a))
        ax.scatter(x,a,s=14,c=color,alpha=.7,linewidths=0)
        ax.plot([i-.27,i+.27],[np.median(a)]*2,color='black',lw=1.8)
    ax.set_xticks(range(7),SHORT,rotation=35,ha='right')
    if threshold is not None:ax.axhline(threshold,color='.45',ls='--',lw=.8)
    if ylim:ax.set_ylim(*ylim)

for old,new in [(1,1),(4,3),(2,'S1')]:
    for ext in ['png','pdf']:shutil.copy2(SRC/'figures'/f'Figure{old}.{ext}',OUT/'figures'/f'Figure{new}.{ext}')

fig,axs=plt.subplots(3,2,figsize=(9.2,10),layout='constrained');a,b,c,e,f,g=axs.flat
fields=pd.read_csv(R/'v23_fem_fullsearch_300/fields_perhead.csv')
for model,color,label in [('sp2hom','#999999','2 mm'),('sp1hom','#0072B2','1 mm')]:
    t=fields[fields.model==model];a.scatter(t.r,t.bias_pct,s=12,alpha=.6,color=color,label=label)
a.axhline(0,color='.5',ls='--',lw=.7);a.set(xlabel='Correlation of basis-field magnitudes',ylabel='Mean-magnitude bias vs FEM (%)')
a.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=8)
title(a,'A','Field shape and amplitude (300 heads)')
a.set_title('A  Field shape and amplitude (300 heads)',loc='left',fontweight='bold',pad=34)
six=pd.read_csv(R/'v23_fem_fullsearch_300/fullsearch_decisions.csv')
for i,roi in enumerate(six.roi.unique()):
    vals=100-six[six.roi==roi].regret_sp1winner_under_fem_pct.to_numpy()
    b.scatter(i+np.linspace(-.22,.22,len(vals)),vals,s=9,color='#0072B2',alpha=.4,linewidths=0)
    b.plot([i-.27,i+.27],[np.median(vals)]*2,color='black',lw=1.5)
b.set_xticks(range(6),['M1','Thalamus','Insula','ACC','Hipp L','Hipp R'],rotation=25,ha='right');b.set_ylim(0,104)
b.axhline(90,color='.4',ls='--',lw=.8);b.set_ylabel('FEM F retention (%)')
title(b,'B','Six targets: unconstrained choice (1 mm)')
c.plot(np.arange(1,164),np.sort(d.retention_p5),color='#0072B2',lw=1.8)
c.axhline(90,color='.4',ls='--',lw=.8);c.set(xlabel='Region rank (ascending)',ylabel='FEM F retention, head-level p5 (%)',ylim=(80,102))
c.text(.04,.92,'161/163 regions ≥90%',transform=c.transAxes,va='top');title(c,'C','Atlas reported choice: focality retention')
e.plot(np.arange(1,164),np.sort(d.retention_ok_and_feasible_pct),color='#009E73',lw=1.8)
e.set(xlabel='Region rank (ascending)',ylabel='Heads meeting both criteria (%)',ylim=(50,102))
e.text(.20,.12,'Median 93.7%; minimum 57.7%\nFEM T floor + ≥90% FEM F retention',transform=e.transAxes,fontsize=8)
title(e,'D','Atlas reported choice: joint nominal success')
groups(f,'sp_choice_fails_fem_floor_pct',ylim=(0,45));f.set_ylabel('Heads failing FEM T floor (%)')
title(f,'E','Target-floor failures across region sets')
pad=pd.read_csv(R/'v23_pad4_check/per_head.csv')
for i,roi in enumerate(pad.target.unique()):
    p=pad[pad.target==roi]
    for shift,col,color in [(-.13,'pad_T_ratio','#D55E00'),(.13,'pad_F_ratio','#0072B2')]:
        lo,med,hi=np.percentile(p[col],[5,50,95]);g.errorbar(med,i+shift,xerr=[[med-lo],[hi-med]],fmt='o',color=color,ms=4,capsize=2)
g.axvline(1,color='.4',ls='--',lw=.8);g.set_yticks(range(6),['M1','Thalamus','Insula','ACC','Hipp L','Hipp R']);g.invert_yaxis()
g.set_xlabel('Four-pad / 19-pad FEM ratio (same mesh)')
g.plot([],[],color='#D55E00',marker='o',label='T');g.plot([],[],color='#0072B2',marker='o',label='F');g.legend(frameon=False,ncol=2,loc='lower right')
title(g,'F','Six fixed montages: pad sensitivity')
save(fig,2)

atlas=pd.concat([pd.read_csv(R/x/'target_summary_ext.csv') for x in ['v23_target_atlas_300','v23_target_atlas_300_ext','v23_target_atlas_300_thal']]).drop_duplicates(['target','model'])
gain=atlas[atlas.model=='sp1v23'].set_index('target').heads_gain_ge_10pct
d['heads_gain_ge_10pct']=d.target.map(gain)
fig,axs=plt.subplots(2,2,figsize=(9.2,7.5),layout='constrained');a,b,c,e=axs.flat
groups(a,'sp1v23__cohort_served_pct',threshold=50,ylim=(0,103));a.set_ylabel('Heads meeting T floor and F criterion (%)')
title(a,'A','SPICEhead cohort montage: nominal coverage')
groups(b,'heads_gain_ge_10pct',threshold=20,ylim=(0,70));b.set_ylabel('Heads gaining ≥10% in F individually (%)')
title(b,'B','Individual vs cohort montage focality')
for i,(pre,label) in enumerate([('cond','Conductivity'),('jit','Placement'),('rot','Cap rotation'),('seg','Segmentation')]):
    lo=d[f'uq_{pre}_T_ratio_p5'].median();hi=d[f'uq_{pre}_T_ratio_p95'].median()
    c.plot([lo,hi],[i,i],lw=4,color='#0072B2',solid_capstyle='round');c.text(hi+.02,i,f'{lo:.2f}–{hi:.2f}',va='center',fontsize=8)
c.axvline(1,color='.5',ls='--',lw=.8);c.set_yticks(range(4),['Conductivity','Placement','Cap rotation','Segmentation']);c.invert_yaxis();c.set_xlim(.45,1.85)
c.set_xlabel('T / nominal T: regional medians of pooled p5–p95');title(c,'C','FEM-selected protocol: SPICEhead sensitivity')
groups(e,'uq_share_runs_retention_ge_90pct',threshold=95,ylim=(0,103));e.set_ylabel('Physical runs retaining ≥90% F (%)')
title(e,'D','FEM-selected protocol: F-only retention')
save(fig,4)

fig,axs=plt.subplots(2,2,figsize=(9.2,7.5),layout='constrained');a,b,c,e=axs.flat
for i,((tier,_),color) in enumerate(zip(SETS,COLORS)):
    z=d[d.tier==tier];x=i+np.linspace(-.21,.21,len(z));sig=z.reported_M0_age_dec_q<.05
    a.scatter(x,z.reported_M0_age_dec_pct,s=18,facecolors=[color if q else 'white' for q in sig],edgecolors=color,lw=.7)
    a.plot([i-.25,i+.25],[z.reported_M0_age_dec_pct.median()]*2,color='black',lw=1.5)
    sig=z.reported_M3_men_q<.05
    b.scatter(z.reported_M0_men_pct,z.reported_M3_men_pct,s=22,facecolors=[color if q else 'white' for q in sig],edgecolors=color,lw=.7)
a.axhline(0,color='.4',ls='--',lw=.8);a.set_xticks(range(7),SHORT,rotation=35,ha='right');a.set_ylabel('Reported-montage T change per decade (%)')
title(a,'A','Regional age associations (age, sex, site)')
b.axhline(0,color='.4',lw=.8);b.plot([-22,2],[-22,2],color='.7',ls='--',lw=.8);b.set_ylim(-12,10)
b.set(xlabel='Men vs women, base model (%)',ylabel='Men vs women, with head size (%)');title(b,'B','Regional sex associations: reported montage')
fixed=pd.read_csv(R/'v23_atlas_review/mixed_model/fixed_effects.csv')
for i,(out,sm,label) in enumerate([('T_wa','M0','Reported: base'),('T_wa','M3','Reported: +head size'),('T_coh','M0','Cohort: base'),('T_coh','M3','Cohort: +head size')]):
    z=fixed[(fixed.outcome==out)&(fixed.model=='sp1v23')&(fixed.size_model==sm)&(fixed.term=='men')].iloc[0]
    est=z['pct_change']
    c.errorbar(est,i,xerr=[[est-z.ci_lo_pct],[z.ci_hi_pct-est]],fmt='o',color='#0072B2' if out=='T_wa' else '#D55E00',capsize=3)
c.set_yticks(range(4),['Reported: base','Reported: +head size','Cohort: base','Cohort: +head size']);c.invert_yaxis();c.axvline(0,color='.5',ls='--',lw=.8);c.set_xlabel('Pooled men vs women difference in T (%, 95% CI)')
title(c,'C','Head-level mixed models: sex association')
site=pd.read_csv(R/'v23_atlas_review/site_size/site_T_summary.csv');site=site[site.outcome.str.contains('sp1v23__T_wa')]
labels=['Base','+ICV','+Skull','+Soft tissue','+CSF','+All four','+Spongy bone']
e.plot(range(7),site.age_pct_median,marker='o',color='#0072B2');e.axhline(0,color='.4',ls='--',lw=.8)
e.set_xticks(range(7),labels,rotation=35,ha='right');e.set_ylabel('Median regional age association (% per decade)')
title(e,'D','Reported montage: anatomy-model sensitivity')
save(fig,5)
print('Saved Figures 1–5 and S1 as PDF/PNG.')
