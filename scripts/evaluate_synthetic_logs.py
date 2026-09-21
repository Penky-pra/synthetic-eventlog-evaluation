"""Descriptive source/synthetic fidelity, tie sensitivity and generator experiments.
Python 3.8+, numpy, pandas, scipy, matplotlib. No patient-level exports are made
by this evaluator. Generator runs are local synthetic files. See README.md.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLS = ['fhir_patient_id','encounter_id','procedure_id','ops_code',
        'period_start','period_end','procedure_performed_date']


def load_log(path, source=False):
    d = pd.read_csv(path, sep=None, engine='python', dtype=str).fillna('')
    missing = sorted(set(COLS) - set(d.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    raw_n = len(d)
    if source:
        d = d.drop_duplicates().copy()
    elif d.duplicated().any():
        raise ValueError('Synthetic file has duplicate rows; do not silently remove them.')
    for c in COLS:
        if d[c].str.strip().eq('').any():
            raise ValueError('Empty required field: ' + c)
    if d.procedure_id.duplicated().any():
        raise ValueError('Conflicting/repeated procedure IDs after exact deduplication.')
    for c in COLS[4:]:
        d[c] = d[c].map(lambda v: pd.to_datetime(v, utc=True, errors='raise'))
        if d[c].isna().any():
            raise ValueError('Missing timestamp: ' + c)
    if not ((d.period_start <= d.procedure_performed_date) &
            (d.procedure_performed_date <= d.period_end)).all():
        raise ValueError('Events outside encounter boundaries.')
    if not d.groupby('encounter_id')[['fhir_patient_id','period_start','period_end']].nunique().eq(1).all().all():
        raise ValueError('Inconsistent encounter assignment/boundaries.')
    d = d.sort_values(['encounter_id','procedure_performed_date','ops_code','procedure_id']).reset_index(drop=True)
    d['offset_h'] = (d.procedure_performed_date-d.period_start).dt.total_seconds()/3600
    return d, {'raw_rows':raw_n,'duplicate_rows_removed':raw_n-len(d)}


def tv(a,b):
    na,nb=sum(a.values()),sum(b.values())
    if not na or not nb:
        return np.nan
    return .5*sum(abs(a.get(k,0)/na-b.get(k,0)/nb) for k in set(a)|set(b))


def ecdf_distance(x,y):
    # KS D only: deliberately no independence-based p-value.
    if not len(x) or not len(y):
        return np.nan
    points=np.union1d(x,y)
    return float(np.max(np.abs(np.searchsorted(np.sort(x),points,side='right')/len(x)-
                               np.searchsorted(np.sort(y),points,side='right')/len(y))))


def extract(d,rng=None):
    variants=Counter(); blocks=Counter(); edges=Counter(); strict=Counter(); rows=[]
    tied_groups=0; tied_cases=0
    for cid,g in d.groupby('encounter_id',sort=False):
        acts=g.ops_code.to_numpy().copy()
        times=g.procedure_performed_date.astype('int64').to_numpy()
        grouped=[]; indices=[]
        for t in np.unique(times):
            ix=np.flatnonzero(times==t); indices.append(ix)
            grouped.append(tuple(sorted(acts[ix])))
        blocks[tuple(grouped)]+=1
        tied_cases+=int(any(len(ix)>1 for ix in indices))
        for ix in indices:
            if len(ix)>1:
                tied_groups+=1
                if rng is not None:
                    acts[ix]=rng.permutation(acts[ix])
        variants[tuple(acts)]+=1
        for j in range(len(acts)-1):
            key=(acts[j],acts[j+1]); h=(times[j+1]-times[j])/3.6e12
            edges[key]+=1
            if h>0: strict[key]+=1
            rows.append((cid,g.fhir_patient_id.iloc[0],key[0],key[1],h))
    transitions=pd.DataFrame(rows,columns=['case','patient','from','to','hours'])
    return {'variants':variants,'blocks':blocks,'edges':edges,'strict':strict,
            'tr':transitions,'tied_groups':tied_groups,'tied_cases':tied_cases}


def describe(d,e):
    sizes=d.groupby('encounter_id').size()
    return dict(events=len(d),patients=d.fhir_patient_id.nunique(),encounters=len(sizes),
                activities=d.ops_code.nunique(),single_event_encounters=int(sizes.eq(1).sum()),
                multi_event_encounters=int(sizes.gt(1).sum()),transitions=len(e['tr']),
                distinct_edges=len(e['edges']),variants=len(e['variants']),
                tied_transition_count=int(e['tr'].hours.eq(0).sum()),
                tied_groups=e['tied_groups'],tied_cases=e['tied_cases'])


def timings(x,y,label):
    x=np.asarray(x,float); y=np.asarray(y,float)
    r={'measure':label,'source_n':len(x),'synthetic_n':len(y)}
    for name,q in [('median',.5),('p95',.95),('p99',.99)]:
        a=float(np.quantile(x,q)) if len(x) else np.nan
        b=float(np.quantile(y,q)) if len(y) else np.nan
        r.update({name+'_source_h':a,name+'_synthetic_h':b,name+'_difference_h':b-a,
                  name+'_relative_error_pct':100*(b-a)/a if a else np.nan})
    r['ks_D']=ecdf_distance(x,y)
    r['wasserstein_h']=float(wasserstein_distance(x,y)) if len(x) and len(y) else np.nan
    return r


def compare(a,b,ea=None,eb=None):
    ea=extract(a) if ea is None else ea; eb=extract(b) if eb is None else eb
    metrics={'activity_TV':tv(Counter(a.ops_code),Counter(b.ops_code)),
             'trace_length_TV':tv(Counter(a.groupby('encounter_id').size()),Counter(b.groupby('encounter_id').size())),
             'dfg_TV':tv(ea['edges'],eb['edges']),
             'positive_gap_adjacent_TV':tv(ea['strict'],eb['strict']),
             'variant_TV':tv(ea['variants'],eb['variants']),
             'timestamp_block_variant_TV':tv(ea['blocks'],eb['blocks'])}
    src_edges=set(ea['edges']); syn_edges=set(eb['edges'])
    metrics['edge_set_jaccard']=len(src_edges&syn_edges)/len(src_edges|syn_edges) if src_edges|syn_edges else np.nan
    metrics['unseen_synthetic_edge_mass']=sum(v for k,v in eb['edges'].items() if k not in src_edges)/sum(eb['edges'].values()) if eb['edges'] else np.nan
    metrics['synthetic_variant_mass_seen_in_source']=sum(v for k,v in eb['variants'].items() if k in ea['variants'])/sum(eb['variants'].values())
    time_rows=[timings(a.offset_h,b.offset_h,'all_event_offset'),
               timings(a.groupby('encounter_id').offset_h.min(),b.groupby('encounter_id').offset_h.min(),'first_event_offset'),
               timings(ea['tr'].hours,eb['tr'].hours,'adjacent_gap')]
    for row in time_rows:
        for k,v in row.items():
            if k!='measure': metrics[row['measure']+'_'+k]=v
    return metrics,time_rows


def transition_table(e):
    rows=[]
    for (p,q),g in e['tr'].groupby(['from','to']):
        rows.append({'from':p,'to':q,'n':len(g),'encounters':g.case.nunique(),
                     'patients':g.patient.nunique(),'tied':int(g.hours.eq(0).sum()),
                     'median_h':g.hours.median(),'p95_h':g.hours.quantile(.95)})
    return pd.DataFrame(rows,columns=['from','to','n','encounters','patients','tied','median_h','p95_h'])


def support_analysis(e,name):
    t=transition_table(e); rows=[]
    for threshold in [5,10,20,30]:
        g=t[t.n>=threshold]
        rho=float(spearmanr(g.median_h,g.p95_h)[0]) if len(g)>1 and g.median_h.nunique()>1 and g.p95_h.nunique()>1 else np.nan
        for rank,r in enumerate(g.sort_values(['p95_h','from','to'],ascending=[False,True,True]).itertuples(index=False),1):
            row=dict(zip(g.columns,r)); row.update(dataset=name,min_events=threshold,p95_rank=rank,median_p95_spearman=rho)
            rows.append(row)
    return rows


def bootstrap_tails(e,name,n,rng):
    # Resample patient clusters independently WITHIN each fixed log. These intervals
    # are conditional on the observed log, not generation uncertainty or paired CI.
    rows=[]
    for (p,q),g in e['tr'].groupby(['from','to']):
        if len(g)<10: continue
        clusters=[z.hours.to_numpy() for _,z in g.groupby('patient')]
        vals=[]
        for _ in range(n):
            sample=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
            vals.append(np.quantile(sample,.95))
        rows.append(dict(dataset=name,from_activity=p,to_activity=q,events=len(g),patients=len(clusters),
                         p95_h=g.hours.quantile(.95),ci_low_h=np.quantile(vals,.025),ci_high_h=np.quantile(vals,.975)))
    return rows


def savefig(fig,path):
    fig.tight_layout()
    fig.savefig(str(path)+'.png',dpi=220,bbox_inches='tight')
    fig.savefig(str(path)+'.pdf',bbox_inches='tight')
    plt.close(fig)


def plots(a,b,metrics,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for d,name,color in [(a,'Source','#666666'),(b,'Synthetic','#176b8f')]:
        x=np.sort(d.offset_h)
        axs[0].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=name,color=color)
        q=np.linspace(.90,.999,100)
        axs[1].plot(q,np.quantile(x,q),label=name,color=color)
    axs[0].set(xlabel='Admission-to-event interval (hours)',ylabel='Empirical cumulative probability',xscale='symlog')
    axs[1].set(xlabel='Quantile',ylabel='Admission-to-event interval (hours)')
    axs[0].legend(); axs[1].legend(); savefig(fig,out/'fidelity_timing')
    keys=['activity_TV','trace_length_TV','dfg_TV','variant_TV','timestamp_block_variant_TV']
    fig,ax=plt.subplots(figsize=(7,3.4))
    ax.barh(['Activity frequencies','Trace lengths','Directly-follows frequencies','Exact variants','Timestamp-block variants'],[metrics[k] for k in keys],color='#176b8f')
    ax.set(xlabel='Total variation distance (0 = identical distributions)',xlim=(0,1));ax.invert_yaxis()
    savefig(fig,out/'fidelity_structure')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True)
    ap.add_argument('--synthetic',required=True)
    ap.add_argument('--out',default='additional_evaluation')
    ap.add_argument('--tie-repeats',type=int,default=100)
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20260917)
    ap.add_argument('--generator-seeds',type=int,default=0,help='0: evaluate existing log only; 30: run both models over 30 seeds')
    ap.add_argument('--first-generation-seed',type=int,default=20260915)
    args=ap.parse_args()
    if args.tie_repeats<1 or args.bootstrap<1 or args.generator_seeds<0: ap.error('Invalid repeat counts')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    a,prep=load_log(args.source,True); b,_=load_log(args.synthetic)
    ea,eb=extract(a),extract(b)
    metrics,time_rows=compare(a,b,ea,eb)
    pd.DataFrame([dict(dataset='source',**describe(a,ea)),dict(dataset='synthetic',**describe(b,eb))]).to_csv(out/'log_summary.csv',index=False)
    pd.DataFrame([metrics]).to_csv(out/'fidelity_metrics.csv',index=False)
    pd.DataFrame(time_rows).to_csv(out/'timing_fidelity.csv',index=False)
    by_activity=[]
    for code in sorted(set(a.ops_code)|set(b.ops_code)):
        by_activity.append(timings(a.loc[a.ops_code==code,'offset_h'],b.loc[b.ops_code==code,'offset_h'],code))
    pd.DataFrame(by_activity).to_csv(out/'activity_timing_fidelity.csv',index=False)
    pd.DataFrame({'source':a.ops_code.value_counts(),'synthetic':b.ops_code.value_counts()}).fillna(0).rename_axis('activity').to_csv(out/'activity_counts.csv')
    ta,tb=transition_table(ea),transition_table(eb)
    ta.merge(tb,on=['from','to'],how='outer',suffixes=('_source','_synthetic')).to_csv(out/'transition_fidelity.csv',index=False)
    pd.DataFrame(support_analysis(ea,'source')+support_analysis(eb,'synthetic')).to_csv(out/'support_sensitivity.csv',index=False)
    rng=np.random.default_rng(args.seed)
    tie_rows=[]
    for i in range(args.tie_repeats):
        ra,rb=extract(a,rng),extract(b,rng)
        tie_rows.append(dict(repeat=i,source_dfg_TV_vs_canonical=tv(ea['edges'],ra['edges']),
                             synthetic_dfg_TV_vs_canonical=tv(eb['edges'],rb['edges']),
                             source_variant_TV_vs_canonical=tv(ea['variants'],ra['variants']),
                             synthetic_variant_TV_vs_canonical=tv(eb['variants'],rb['variants']),
                             source_synthetic_dfg_TV=tv(ra['edges'],rb['edges']),
                             source_synthetic_variant_TV=tv(ra['variants'],rb['variants'])))
    pd.DataFrame(tie_rows).to_csv(out/'tie_order_sensitivity.csv',index=False)
    pd.DataFrame(bootstrap_tails(ea,'source',args.bootstrap,rng)+bootstrap_tails(eb,'synthetic',args.bootstrap,rng)).to_csv(out/'transition_p95_cluster_bootstrap.csv',index=False)
    plots(a,b,metrics,out)
    manifest={'source_preparation':prep,'python':platform.python_version(),'numpy':np.__version__,
              'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
              'arguments':vars(args),'input_sha256':{label:hashlib.sha256(Path(path).read_bytes()).hexdigest() for label,path in [('source',args.source),('synthetic',args.synthetic)]},
              'interpretation':'Development-sample descriptive fidelity, not independent validation, clinical validity, or privacy certification.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.generator_seeds:
        generator=Path(__file__).with_name('generate_synthetic_evaluation.py')
        if not generator.exists(): raise FileNotFoundError('Keep generate_synthetic_evaluation.py next to this script.')
        rows=[]; diagnostics=[]
        for seed in range(args.first_generation_seed,args.first_generation_seed+args.generator_seeds):
            for model in ['markov','independent']:
                run=out/'runs'/('{}_{:d}'.format(model,seed)); run.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,str(generator),'--input',str(Path(args.source).resolve()),'--out',str(run),'--seed',str(seed),'--activity-model',model]
                with (run/'generator.log').open('w',encoding='utf-8') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                diag=json.loads((run/'validation.json').read_text())
                diagnostics.append(dict(model=model,seed=seed,pre_shortened=diag['synthetic_encounters_shortened_to_prevent_overlap'],post_shortened=diag['post_calibration_encounters_shortened'],checks_passed=sum(diag['structural_checks'].values())))
                for stage,file in [('pre','pre_calibration.csv'),('post','new_dataset_synthetic.csv')]:
                    s,_=load_log(run/file); m,_=compare(a,s,ea)
                    rows.append(dict(model=model,stage=stage,seed=seed,**m))
                pd.DataFrame(rows).to_csv(out/'generation_runs.csv',index=False)
                pd.DataFrame(diagnostics).to_csv(out/'generation_diagnostics.csv',index=False)
                print('Completed model={} seed={}'.format(model,seed),flush=True)
        all_runs=pd.DataFrame(rows)
        summaries=[]
        for (model,stage),g in all_runs.groupby(['model','stage']):
            for measure in metrics:
                values=g[measure].dropna()
                summaries.append(dict(model=model,stage=stage,measure=measure,seeds=len(values),mean=values.mean(),sd=values.std(),median=values.median(),p025=values.quantile(.025),p975=values.quantile(.975)))
        pd.DataFrame(summaries).to_csv(out/'generation_summary.csv',index=False)
        fig,axs=plt.subplots(1,3,figsize=(11,3.8))
        order=[('independent','pre'),('independent','post'),('markov','pre'),('markov','post')]
        for ax,key,title in zip(axs,['dfg_TV','variant_TV','all_event_offset_wasserstein_h'],['DFG total variation','Variant total variation','Timing Wasserstein (hours)']):
            data=[all_runs.loc[(all_runs.model==m)&(all_runs.stage==s),key].to_numpy() for m,s in order]
            ax.boxplot(data); ax.set_xticks(range(1,5));ax.set_xticklabels(['Ind.\npre','Ind.\npost','Markov\npre','Markov\npost']);ax.set_title(title)
        savefig(fig,out/'generation_comparison')
    print('Evaluation complete: '+str(out.resolve()))


if __name__=='__main__':
    main()
