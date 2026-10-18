"""Repeated nested subject cross-validation using EXISTING EEG features.

Colab: upload this file; then run:
    from eeg_cross_validation import run_cached_cv
    cv_results = run_cached_cv(data, root=lab.ROOT)

No model imports, encoder execution, downloads, or GPU are required.
Use the same unchanged data and feature row ordering from the original run.
"""
from pathlib import Path
import csv
import hashlib
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

METHODS = ('cbramod_pretrained','cbramod_random','labram_pretrained','labram_random')
C_VALUES = (.01,.1,1.,10.)


def _participant_scores(y, predictions, probabilities, groups):
    """Equal participant weights; every participant must have both labels."""
    scores=[]
    for subject in np.unique(groups):
        mask=groups==subject
        if len(np.unique(y[mask]))!=2:
            raise ValueError(f'Participant {subject} must have both labels.')
        scores.append(dict(subject=int(subject), windows=int(mask.sum()),
            balanced_accuracy=float(balanced_accuracy_score(y[mask],predictions[mask])),
            macro_f1=float(f1_score(y[mask],predictions[mask],average='macro',zero_division=0)),
            auroc=float(roc_auc_score(y[mask],probabilities[mask]))))
    return scores


def _splits(groups,n_splits,seed):
    """KFold on unique participant IDs, then map IDs back to windows."""
    people=np.unique(groups)
    if not 2<=n_splits<=len(people):
        raise ValueError('Fold count must be between two and the participant count.')
    for train_ids,test_ids in KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(people):
        train=np.flatnonzero(np.isin(groups,people[train_ids]))
        test=np.flatnonzero(np.isin(groups,people[test_ids]))
        assert not set(groups[train]) & set(groups[test])
        yield train,test


def _classifier(c):
    # The scaler is fitted inside each inner training fold, never beforehand.
    return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,
                         class_weight='balanced',solver='lbfgs',random_state=42))


def _select_c(features,y,groups,inner_splits):
    """Receives ONLY outer-training rows; cannot inspect outer-test labels."""
    candidates=[]
    for c in C_VALUES:
        subject_scores=[]
        for train,val in inner_splits:
            model=_classifier(c)
            model.fit(features[train],y[train])
            predictions=model.predict(features[val])
            # Select on equal-weight participant balanced accuracy.
            subject_scores.extend(float(balanced_accuracy_score(y[val][groups[val]==s],
                                      predictions[groups[val]==s])) for s in np.unique(groups[val]))
        candidates.append(dict(C=float(c),inner_subject_balanced_accuracy=float(np.mean(subject_scores))))
    # Stable tie break prefers the smaller C (stronger regularization).
    selected=max(candidates,key=lambda item:item['inner_subject_balanced_accuracy'])
    return selected['C'],candidates


def evaluate_representations(features,y,subjects,*,repeats=3,outer_folds=5,inner_folds=4,
                             seed=42,control_labels=None,output_dir=None,verbose=True):
    """Shared nested participant splits for all representations.

    Freeze this protocol before running. Repetition SD is descriptive split
    variability, not a standard error or confidence interval: people recur.
    """
    y=np.asarray(y,dtype=int)
    subjects=np.asarray(subjects)
    if y.ndim!=1 or subjects.shape!=y.shape or not np.array_equal(np.unique(y),[0,1]):
        raise ValueError('Expected 1D binary labels and matching participant IDs.')
    if not isinstance(repeats,int) or repeats<1:
        raise ValueError('repeats must be a positive integer.')
    for name,values in features.items():
        if values.ndim!=2 or len(values)!=len(y) or not np.isfinite(values).all():
            raise ValueError(f'Invalid features for {name}.')
    labels={name:y for name in features}
    if control_labels is not None:
        if 'bandpower_label_control' not in features:
            raise ValueError('Control labels supplied without control representation.')
        labels['bandpower_label_control']=np.asarray(control_labels,dtype=int)
    # Require both labels per person to make subject-level AUROC well-defined.
    for name,targets in labels.items():
        if targets.shape!=y.shape:
            raise ValueError(f'Label shape mismatch for {name}.')
        for person in np.unique(subjects):
            if not np.array_equal(np.unique(targets[subjects==person]),[0,1]):
                raise ValueError(f'Both classes required for participant {person} ({name}).')
    probabilities={name:np.full((repeats,len(y)),np.nan) for name in features}
    predictions={name:np.full((repeats,len(y)),-1,dtype=int) for name in features}
    report=dict(protocol=dict(repeats=repeats,outer_folds=outer_folds,inner_folds=inner_folds,
               seed=seed,C_values=list(C_VALUES),selection='mean participant balanced accuracy',
               threshold='logistic regression default; no test-set threshold tuning',
               weighting='equal participants',frozen_encoders=True),
               caveats=[
                   'LaBraM pretraining includes this PhysioNet corpus. Supervised grouping does not remove pretraining exposure.',
                   'This is a follow-up analysis after inspecting an earlier split, not a fresh untouched test.',
                   'Repeat SD is descriptive; repeats and windows are not independent participants.',
                   'Random backbones use the single existing initialization, not multiple initialization seeds.',
                   'One subject-level label permutation is a sanity check, not a significance test.'],
               folds=[],participant_scores=[],summary={},paired_pretraining_differences={})
    for repeat in range(repeats):
        for fold,(train,test) in enumerate(_splits(subjects,outer_folds,seed+repeat)):
            inner=list(_splits(subjects[train],inner_folds,seed+1000+repeat*outer_folds+fold))
            inner_ids=[]
            for inner_train,inner_val in inner:
                assert not set(subjects[train][inner_train]) & set(subjects[test])
                assert not set(subjects[train][inner_val]) & set(subjects[test])
                inner_ids.append(dict(train=np.unique(subjects[train][inner_train]).tolist(),
                                      validation=np.unique(subjects[train][inner_val]).tolist()))
            fold_result=dict(repeat=repeat+1,fold=fold+1,
                             train_subjects=np.unique(subjects[train]).tolist(),
                             test_subjects=np.unique(subjects[test]).tolist(),
                             inner_subject_splits=inner_ids,methods={})
            for name,values in features.items():
                if verbose:
                    print(f'Repeat {repeat+1}/{repeats}, fold {fold+1}/{outer_folds}: {name}',flush=True)
                targets=labels[name]
                c,grid=_select_c(values[train],targets[train],subjects[train],inner)
                model=_classifier(c)
                model.fit(values[train],targets[train])
                predicted=model.predict(values[test])
                probability=model.predict_proba(values[test])[:,1]
                predictions[name][repeat,test]=predicted
                probabilities[name][repeat,test]=probability
                scores=_participant_scores(targets[test],predicted,probability,subjects[test])
                fold_result['methods'][name]=dict(selected_C=c,inner_grid=grid,
                    test_mean_participant_balanced_accuracy=float(np.mean([s['balanced_accuracy'] for s in scores])))
                for row in scores:
                    report['participant_scores'].append(dict(repeat=repeat+1,fold=fold+1,method=name,**row))
            report['folds'].append(fold_result)
            if output_dir is not None:
                out=Path(output_dir)
                out.mkdir(parents=True,exist_ok=True)
                (out/'cv_progress.json').write_text(json.dumps(report,indent=2))
    for name in features:
        if not np.isfinite(probabilities[name]).all() or (predictions[name]<0).any():
            raise RuntimeError('Some windows never received an out-of-fold prediction.')
        rows=[r for r in report['participant_scores'] if r['method']==name]
        method_summary={}
        for metric in ['balanced_accuracy','macro_f1','auroc']:
            per_repeat=[float(np.mean([r[metric] for r in rows if r['repeat']==rep+1])) for rep in range(repeats)]
            method_summary[metric]=dict(mean=float(np.mean(per_repeat)),
                repeat_sd=float(np.std(per_repeat,ddof=1)) if repeats>1 else 0.,per_repeat=per_repeat)
        report['summary'][name]=method_summary
    for kind in ['cbramod','labram']:
        pre,rand=kind+'_pretrained',kind+'_random'
        if pre not in features or rand not in features:
            continue
        differences={}
        for person in np.unique(subjects):
            def average(method):
                return np.mean([r['balanced_accuracy'] for r in report['participant_scores']
                                if r['subject']==person and r['method']==method])
            differences[str(person)]=float(average(pre)-average(rand))
        report['paired_pretraining_differences'][kind]=dict(
            mean_percentage_points=float(100*np.mean(list(differences.values()))),
            positive_participants=sum(v>0 for v in differences.values()),
            total_participants=len(differences),per_participant=differences)
    if output_dir is not None:
        _write_outputs(report,labels,subjects,predictions,probabilities,Path(output_dir))
    return report


def _write_csv(path,rows):
    if not rows:
        return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(report,labels,subjects,predictions,probabilities,out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'cv_results.json').write_text(json.dumps(report,indent=2))
    _write_csv(out/'cv_participants.csv',report['participant_scores'])
    summary_rows=[]
    for name,metrics in report['summary'].items():
        row={'method':name}
        for metric,values in metrics.items():
            row[metric+'_mean']=values['mean']
            row[metric+'_repeat_sd']=values['repeat_sd']
        summary_rows.append(row)
    _write_csv(out/'cv_summary.csv',summary_rows)
    arrays={'subjects':subjects}
    for name in predictions:
        arrays[name+'_labels']=labels[name]
        arrays[name+'_predictions']=predictions[name]
        arrays[name+'_probabilities']=probabilities[name]
    np.savez_compressed(out/'cv_oof_predictions.npz',**arrays)


def _fingerprint(values):
    contiguous=np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest()


def run_cached_cv(data,root='.',*,repeats=3,outer_folds=5,inner_folds=4,seed=42):
    """Notebook entrypoint. Requires all four previously saved model feature files."""
    import eeg_lab as lab
    root=Path(root)
    X,y,subjects=data['X'],data['y'],data['subjects']
    if X.ndim!=3 or X.shape[1:]!=(19,800) or len(X)!=len(y):
        raise ValueError('Expected unchanged data from the original 19-channel experiment.')
    # Where available, compare against the dataset saved during preparation.
    # Legacy feature caches have no row fingerprint: keep their data unchanged.
    if (root/'dataset.npz').exists():
        with np.load(root/'dataset.npz',allow_pickle=False) as saved:
            for key in ['X','y','subjects']:
                if not np.array_equal(saved[key],data[key]):
                    raise ValueError(f'{key} differs from saved dataset.npz; do not reuse these feature caches.')
    features={'bandpower':lab.bandpower_features(X)}
    manifest=dict(data_fingerprints={k:_fingerprint(data[k]) for k in ['X','y','subjects']},
                  feature_sha256={},row_order_note='Legacy caches assumed to match original unchanged data.')
    for name in METHODS:
        path=root/f'{name}_features.npy'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path.name}. Finish the original extraction cell first.')
        values=np.load(path,allow_pickle=False)
        if values.shape!=(len(y),200):
            raise ValueError(f'Unexpected shape {values.shape} in {path.name}.')
        features[name]=values
        manifest['feature_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    # Preserve recordings: each subject's two condition labels are either both
    # retained or swapped. Use a single fixed permutation as a diagnostic only.
    rng=np.random.default_rng(seed)
    flipped={s:bool(rng.integers(2)) for s in np.unique(subjects)}
    control=np.array([int(label)^int(flipped[s]) for label,s in zip(y,subjects)])
    features['bandpower_label_control']=features['bandpower']
    out=root/'cross_validation'
    out.mkdir(parents=True,exist_ok=True)
    (out/'cv_input_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Using cached encoder features. No GPU work or downloads.',flush=True)
    print('This is a follow-up analysis of the same twenty people; pretraining overlap still applies.',flush=True)
    # Limit CPU thread oversubscription, especially in hosted notebook runtimes.
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        report=evaluate_representations(features,y,subjects,repeats=repeats,outer_folds=outer_folds,
                    inner_folds=inner_folds,seed=seed,control_labels=control,output_dir=out)
    print('\nMean participant metrics across repeated outer cross-validation:')
    for name,metrics in report['summary'].items():
        ba=metrics['balanced_accuracy']
        print(f"{name:26s} BA {100*ba['mean']:5.1f}% (repeat SD {100*ba['repeat_sd']:.1f} pp)"
              f" | F1 {metrics['macro_f1']['mean']:.3f} | AUROC {metrics['auroc']['mean']:.3f}")
    print('\nPretrained minus random (paired by participant):')
    for name,comparison in report['paired_pretraining_differences'].items():
        print(f"{name}: {comparison['mean_percentage_points']:+.2f} percentage points;")
        print(f"  positive for {comparison['positive_participants']}/{comparison['total_participants']} participants")
    print('\nOutputs saved in:',out)
    plot_cv(report,out)
    return report


def plot_cv(report,output_dir=None):
    import matplotlib.pyplot as plt
    names=list(report['summary'])
    x=np.arange(len(names))
    means=[100*report['summary'][n]['balanced_accuracy']['mean'] for n in names]
    sd=[100*report['summary'][n]['balanced_accuracy']['repeat_sd'] for n in names]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(x,means,yerr=sd,capsize=4,color='#397d96')
    for i,name in enumerate(names):
        people=sorted({r['subject'] for r in report['participant_scores'] if r['method']==name})
        scores=[100*np.mean([r['balanced_accuracy'] for r in report['participant_scores']
                  if r['method']==name and r['subject']==s]) for s in people]
        offset=np.linspace(-.22,.22,len(scores))
        ax.scatter(i+offset,scores,color='black',alpha=.45,s=13)
    ax.axhline(50,color='gray',linestyle='--')
    ax.set_ylim(0,105)
    ax.set_xticks(x,names,rotation=25,ha='right')
    ax.set_ylabel('Mean participant balanced accuracy (%)')
    ax.set_title('Dots: participants averaged over repeats; error bars: descriptive repeat SD')
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(Path(output_dir)/'cv_comparison.png',dpi=160)
    plt.show()
