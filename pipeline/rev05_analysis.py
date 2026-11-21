"""Reproduce the locally recoverable v23 revision calculations; never run forward solves.

Run with a Python environment containing numpy and pandas. Source scientific outputs
remain read-only. BH families are 163 regions per outcome/model/covariate set/term.
"""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'Paper_A_SPICEhead/Brain Stimulation'
SRC = J / 'Rev0.4_2026-10-30'
OUT = J / 'Rev0.5_2026-11-02'
EV = OUT / 'evidence_revision'
R = ROOT / 'results_v23'
SETS = [('tier1','Cerebral (AAL)'),('tier2','Psychiatric'),('ref6','Reference spheres'),
        ('cerebellum','Cerebellum'),('brainstem','Brainstem nuclei'),
        ('thalamus','Thalamic nuclei'),('cingulate','Supracallosal ACC')]

def bh(p):
    p=np.asarray(p,float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    order=np.argsort(p); q=np.empty(len(p))
    q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
    return q

def run():
    EV.mkdir(parents=True,exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    manifest={str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(SRC.rglob('*')) if p.is_file()}
    mp=EV/'rev04_baseline_sha256.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest, 'Baseline changed'
    else: mp.write_text(json.dumps(manifest,indent=2))
    for name in ['evidence_v23','references']:
        if not (OUT/name).exists(): shutil.copytree(SRC/name,OUT/name)
    sources=['v23_atlas_review/mixed_model/per_region_vs_mixed.csv',
             'v23_atlas_review/mixed_model/fixed_effects.csv',
             'v23_atlas_review/mixed_model/variance_components.csv',
             'v23_atlas_review/region_review.csv','v23_atlas_parameters/target_parameters.csv',
             'v23_uq/atlas_300/per_target.csv','v23_uq/atlas_thal_300/per_target.csv',
             'v23_uq/mask_empty_300.csv','v23_atlas_review/site_size/site_T_summary.csv',
             'v23_atlas_review/site_size/site_T_regions.csv','v23_pad4_check/per_head.csv',
             'v23_fem_fullsearch_300/fields_perhead.csv','v23_fem_fullsearch_300/fullsearch_decisions.csv']
    (EV/'source_sha256.json').write_text(json.dumps({p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sources},indent=2))
    d=pd.read_csv(R/sources[0]); rv=pd.read_csv(R/'v23_atlas_review/region_review.csv')
    rv['region']=rv.target.str.removeprefix('ref6:').str.removeprefix('brainstem:')
    # Reference targets retain ref6: in the mixed-model output; avoid anatomy-label collisions.
    names=set(d.region)
    rv['region']=[t if t in names else t.split(':')[-1] for t in rv.target]
    assert rv.region.nunique()==163 and set(rv.region)==names
    d['q_bh163']=d.groupby(['outcome','model','size_model','term']).ols_p.transform(lambda p:bh(p))
    assert (d.groupby(['outcome','model','size_model','term']).size()==163).all()
    d=d.merge(rv[['region','target','tier']],on='region',validate='many_to_one')
    d.to_csv(EV/'demographics_reported_and_cohort_bh.csv',index=False)
    stats=[]
    for k,g in d.groupby(['outcome','model','size_model','term']):
        stats.append(dict(zip(['outcome','model','size_model','term'],k),regions=len(g),
                          median_pct=float(g.ols_pct.median()),
                          negative_q05=int(((g.q_bh163<.05)&(g.ols_pct<0)).sum()),
                          positive_q05=int(((g.q_bh163<.05)&(g.ols_pct>0)).sum())))
    pd.DataFrame(stats).to_csv(EV/'demographic_summary.csv',index=False)
    # Build a single primary regional table for all revised prose, tables, figures and workbook columns.
    primary=rv.copy()
    for out,label in [('T_wa','reported'),('T_coh','cohort')]:
        for size in ['M0','M3']:
            for term in ['age_dec','men']:
                g=d[(d.outcome==out)&(d.model=='sp1v23')&(d.size_model==size)&(d.term==term)]
                prefix=f'{label}_{size}_{term}'
                primary=primary.merge(g[['target','ols_pct','ols_p','q_bh163']].rename(columns={
                    'ols_pct':prefix+'_pct','ols_p':prefix+'_p','q_bh163':prefix+'_q'}),on='target',validate='one_to_one')
    params=pd.read_csv(R/'v23_atlas_parameters/target_parameters.csv')
    for model,label in [('sp1v23','SPICEhead'),('fem','FEM')]:
        primary=primary.merge(params[params.model==model][['target','cohort_choice']].rename(
            columns={'cohort_choice':label+'_cohort_montage'}),on='target',validate='one_to_one')
    primary['UQ_protocol_selection_model']='FEM'
    primary['UQ_field_model']='SPICEhead v23 1 mm'
    primary['UQ_protocol_matches_SPICEhead']=(primary.SPICEhead_cohort_montage==primary.FEM_cohort_montage)
    uq=pd.concat([pd.read_csv(R/'v23_uq'/x/'per_target.csv') for x in ['atlas_300','atlas_thal_300']]).drop_duplicates('target',keep='first')
    assert len(uq)==163 and (uq.heads==300).all()
    # All mask exposure summaries are withheld: zeros/nonfinite handling differs by target set.
    uq=uq.drop(columns=[c for c in uq if c.startswith('mask_')])
    uq=uq.rename(columns={c:'uq_'+c for c in uq if c not in ['target','tier']}).drop(columns='tier')
    primary=primary.merge(uq,on='target',validate='one_to_one')
    mask=pd.read_csv(R/'v23_uq/mask_empty_300.csv')
    assert set(mask.target)==set(primary.target)
    mask['erosion_unavailable_heads']=mask.mask_erode_nonfinite+mask.mask_erode_zero
    mask['dilation_unavailable_heads']=mask.mask_dilate_nonfinite+mask.mask_dilate_zero
    mask['mask_assessed_heads']=mask.heads
    mask['erosion_nonempty_heads']=mask.heads-mask.erosion_unavailable_heads
    mask['dilation_nonempty_heads']=mask.heads-mask.dilation_unavailable_heads
    # Nonempty is not the >=5 voxel validity denominator; this remains unknown locally.
    primary=primary.merge(mask.drop(columns='heads'),on='target',validate='one_to_one')
    primary['mask_corrected_exposure_status']='Unavailable: per-run recollection required'
    primary['UQ_joint_success_status']='Unavailable: per-run T/Tmax and F/Fconstrained required'
    primary.to_csv(EV/'regional_revision_data.csv',index=False)
    (EV/'workbook_revision_data.json').write_text(json.dumps(
        primary.astype(object).where(pd.notna(primary),None).to_dict(orient='records'),
        allow_nan=False),encoding='utf-8')
    primary[~primary.UQ_protocol_matches_SPICEhead][['target','tier','SPICEhead_cohort_montage','FEM_cohort_montage']].to_csv(EV/'protocol_mismatches_34.csv',index=False)
    mask[mask.erosion_unavailable_heads>0].to_csv(EV/'mask_failures_10.csv',index=False)
    v=pd.read_csv(R/'v23_atlas_review/mixed_model/variance_components.csv')
    v=v[(v.outcome=='T_wa')&(v.model=='sp1v23')].set_index('size_model')
    counts=dict(regions=163,heads=300,mismatched_protocols=int((~primary.UQ_protocol_matches_SPICEhead).sum()),
                mask_affected_regions=int((primary.erosion_unavailable_heads>0).sum()),
                reported_age_negative_q05=int(((primary.reported_M0_age_dec_q<.05)&(primary.reported_M0_age_dec_pct<0)).sum()),
                reported_age_positive_q05=int(((primary.reported_M0_age_dec_q<.05)&(primary.reported_M0_age_dec_pct>0)).sum()),
                reported_M3_sex_q05=int((primary.reported_M3_men_q<.05).sum()),
                fem_joint_nominal_median_pct=float(primary.retention_ok_and_feasible_pct.median()),
                fem_joint_nominal_min_pct=float(primary.retention_ok_and_feasible_pct.min()),
                sp_cohort_served_median_pct=float(primary.sp1v23__cohort_served_pct.median()),
                sp_cohort_served_below50_regions=int((primary.sp1v23__cohort_served_pct<50).sum()),
                uq_F_only_robust_regions=int(primary.uq_fixed_protocol_robust.sum()),
                between_head_variance_reduction_pct=float(100*(1-(v.loc['M3','sd_head']/v.loc['M0','sd_head'])**2)),
                physical_perturbations_per_head=74,total_evaluations_per_head=77)
    (EV/'local_calculation_summary.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts,indent=2))
    return primary,d

if __name__=='__main__': run()
