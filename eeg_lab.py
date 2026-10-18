"""Small, subject-disjoint EEG transfer-learning experiment.

Only NumPy/SciPy/scikit-learn are needed for the offline smoke test.
Model code and checkpoints are downloaded from pinned upstream revisions.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import urllib.request

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CB_REV = 'b9e961003214326972c567eff390e75b0287e32a'
LA_REV = 'c431221e6cfd23dbfa9950e0180682fb322b0548'
# Match the spatial ordering of CBraMod's pretraining montage. Modern names
# T7/T8/P7/P8 correspond to its older T3/T4/T5/T6 electrode names.
CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8',
            'P7','P3','Pz','P4','P8','O1','O2']
BANDS = [(1,4),(4,8),(8,13),(13,30),(30,40)]


def download(url, destination):
    """Write atomically; interrupted downloads are never accepted as cached files."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.partial')
        try:
            with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as out:
                while block := source.read(1024 * 1024):
                    out.write(block)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def setup_assets():
    """Fetch small official source files plus the two pretrained backbones."""
    base = ROOT / 'upstream'
    sources = [
        ('wjq-learning/CBraMod', CB_REV, 'CBraMod',
         ['models/cbramod.py','models/criss_cross_transformer.py','LICENSE']),
        ('935963004/LaBraM', LA_REV, 'LaBraM',
         ['modeling_finetune.py','utils.py','LICENSE'])]
    for repo, revision, folder, files in sources:
        for name in files:
            download(f'https://raw.githubusercontent.com/{repo}/{revision}/{name}',
                     base / folder / name)
    download('https://huggingface.co/weighting666/CBraMod/resolve/500543c/pretrained_weights.pth',
             ROOT / 'weights/cbramod.pth')
    download(f'https://raw.githubusercontent.com/935963004/LaBraM/{LA_REV}/checkpoints/labram-base.pth',
             ROOT / 'weights/labram-base.pth')
    info = {'cbramod_source_revision': CB_REV, 'labram_source_revision': LA_REV,
            'cbramod_checkpoint_revision': '500543c'}
    for name in ['cbramod.pth','labram-base.pth']:
        info[name + '_sha256'] = hashlib.sha256((ROOT / 'weights' / name).read_bytes()).hexdigest()
    (ROOT / 'asset_manifest.json').write_text(json.dumps(info, indent=2))
    print('Sources and weights ready. Revisions and weight hashes: asset_manifest.json')


def load_public_data(subjects=range(1,21)):
    """Runs 1/2: eyes open/closed; 4 seconds, 19 channels, 200 Hz.

    MNE reads EDF in volts. Divide by 100 microvolts (1e-4 V) for
    pretrained-model inputs. Keep a fixed amplitude scale, not per-window z-scores.
    """
    import mne
    from mne.datasets import eegbci
    cache = ROOT / 'data'
    cache.mkdir(exist_ok=True)
    xs, ys, groups, recordings, starts, counts = [], [], [], [], [], []
    for subject in subjects:
        for run, label in [(1,0),(2,1)]:
            paths = eegbci.load_data(subject, [run], path=str(cache), update_path=False, verbose=False)
            raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.pick(CHANNELS)
            raw.reorder_channels(CHANNELS)
            # Common average computed over the selected 19 channels per recording.
            raw.set_eeg_reference('average', projection=False, verbose=False)
            # This is our shared downstream preprocessing, not an exact paper replication.
            raw.filter(1.,40., method='iir', verbose=False)
            raw.resample(200, verbose=False)
            signal = raw.get_data() / 1e-4
            accepted = 0
            # Trim both ends, use complete non-overlapping windows only.
            for start in range(2*200, signal.shape[1] - 2*200 - 800 + 1, 800):
                window = signal[:, start:start+800]
                # A generous fixed artifact threshold: 500 microvolts.
                if not np.isfinite(window).all() or np.max(np.abs(window)) > 5:
                    continue
                xs.append(window.astype('float32'))
                ys.append(label)
                groups.append(subject)
                recordings.append(f'S{subject:03d}R{run:02d}')
                starts.append(start / 200)
                accepted += 1
            counts.append({'subject':subject,'run':run,'label':label,'windows':accepted})
    if not xs:
        raise RuntimeError('No valid windows. Inspect data and preprocessing.')
    result = dict(X=np.stack(xs), y=np.array(ys), subjects=np.array(groups),
                  recordings=np.array(recordings), start_seconds=np.array(starts),
                  channels=np.array(CHANNELS), sample_rate=np.array(200))
    np.savez_compressed(ROOT / 'dataset.npz', **result)
    (ROOT / 'data_counts.json').write_text(json.dumps(counts, indent=2))
    return result


def split_subjects(subjects, seed=42):
    """60/20/20 split, shared across every representation and control."""
    ids = np.random.default_rng(seed).permutation(np.unique(subjects))
    if len(ids) < 5:
        raise ValueError('Use at least five subjects; twenty is the default.')
    n_train, n_val = int(.6*len(ids)), max(1,int(.2*len(ids)))
    partitions = [ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]]
    assert all(len(p) for p in partitions)
    masks = [np.isin(subjects, part) for part in partitions]
    assert not any(set(partitions[i]) & set(partitions[j]) for i in range(3) for j in range(i))
    return masks, partitions


def bandpower_features(X, sfreq=200):
    """Five log bandpowers per channel; no learned transformation here."""
    frequencies, psd = welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
    df = frequencies[1] - frequencies[0]
    powers = [np.log10(np.maximum(psd[..., (frequencies>=lo)&(frequencies<hi)].sum(-1)*df, 1e-20))
              for lo,hi in BANDS]
    return np.stack(powers, axis=-1).reshape(len(X), -1)


def patchify(X):
    if X.ndim != 3 or X.shape[-1] != 800:
        raise ValueError('Expected (windows, channels, 800) at 200 Hz.')
    return np.ascontiguousarray(X.reshape(len(X), X.shape[1], 4, 200))


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _channel_indices():
    # Read the official channel table without importing utils.py, which pulls
    # in unrelated training packages. Its first index is reserved for CLS.
    tree = ast.parse((ROOT / 'upstream/LaBraM/utils.py').read_text())
    table = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == 'standard_1020' for t in node.targets))
    return [0] + [table.index(name.upper())+1 for name in CHANNELS]


def _state_dict(path):
    import argparse
    import torch
    # Released training checkpoints can include NumPy scalar metadata.
    # Permit these specific classes while retaining weights-only loading.
    numpy_scalar = __import__('importlib').import_module('numpy.core.multiarray').scalar
    allowed = [argparse.Namespace, np.dtype, numpy_scalar,
               type(np.dtype('float64')), type(np.dtype('float32'))]
    # NumPy 2 moved the implementation; recent PyTorch can register the
    # legacy checkpoint name explicitly. Pinned local NumPy 1 needs no alias.
    if numpy_scalar.__module__ != 'numpy.core.multiarray':
        allowed.append((numpy_scalar, 'numpy.core.multiarray.scalar'))
    with torch.serialization.safe_globals(allowed):
        data = torch.load(path, map_location='cpu', weights_only=True)
    for key in ['model','state_dict','module']:
        if isinstance(data, dict) and key in data and isinstance(data[key], dict):
            data = data[key]
            break
    data = {k.removeprefix('module.'):v for k,v in data.items()}
    # Released LaBraM pretraining states may wrap the backbone as student.
    if any(k.startswith('student.') for k in data):
        data = {k[len('student.'):]:v for k,v in data.items() if k.startswith('student.')}
    return data


def build_backbone(kind, pretrained=True, seed=42):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    if kind == 'cbramod':
        path = str(ROOT / 'upstream/CBraMod')
        if path not in sys.path:
            sys.path.insert(0, path)
        from models.cbramod import CBraMod
        model = CBraMod()
        if pretrained:
            model.load_state_dict(_state_dict(ROOT / 'weights/cbramod.pth'), strict=True)
        model.proj_out = nn.Identity()
    elif kind == 'labram':
        module = _load_module('official_labram_finetune', ROOT / 'upstream/LaBraM/modeling_finetune.py')
        state = _state_dict(ROOT / 'weights/labram-base.pth') if pretrained else None
        # Pretraining has a final token LayerNorm. Retain it, then mean-pool
        # patch tokens ourselves, avoiding an untrained downstream fc_norm.
        has_layer_scale = state is None or any(k.endswith('gamma_1') for k in state)
        model = module.labram_base_patch200_200(num_classes=0, use_mean_pooling=False,
                    init_values=.1 if has_layer_scale else None, qkv_bias=False)
        if state is not None:
            allowed = ('mask_token','lm_head.','head.')
            state = {k:v for k,v in state.items() if not k.startswith(allowed)}
            # Fail loudly rather than claiming a partially random backbone is pretrained.
            model.load_state_dict(state, strict=True)
    else:
        raise ValueError(kind)
    model.eval()
    model.requires_grad_(False)
    return model


def extract_features(X, kind, pretrained=True, batch_size=8, seed=42, device=None):
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_backbone(kind, pretrained, seed).to(device)
    patches = patchify(X)
    channels = _channel_indices() if kind == 'labram' else None
    features = []
    with torch.inference_mode():
        for start in range(0,len(X),batch_size):
            batch = torch.from_numpy(patches[start:start+batch_size]).to(device)
            if kind == 'cbramod':
                output = model(batch).mean(dim=(1,2))
            else:
                output = model.forward_features(batch, input_chans=channels,
                                               return_patch_tokens=True).mean(dim=1)
            features.append(output.cpu().numpy())
    result = np.concatenate(features)
    if result.shape != (len(X),200) or not np.isfinite(result).all():
        raise RuntimeError(f'Unexpected {kind} features: {result.shape}')
    return result


def evaluate(features, y, subjects, masks):
    train, val, test = masks
    for mask in masks:
        if len(np.unique(y[mask])) != 2:
            raise ValueError('Every split must retain both labels. Inspect data_counts.json.')
    # scaler and classifier see training subjects only; C uses validation only.
    best = None
    for c in [.01,.1,1.,10.]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000,
                                  class_weight='balanced', random_state=42))
        model.fit(features[train], y[train])
        score = balanced_accuracy_score(y[val], model.predict(features[val]))
        if best is None or score > best[0]:
            best = (score, c, model)
    score, c, model = best
    predicted = model.predict(features[test])
    probabilities = model.predict_proba(features[test])[:,1]
    per_subject = {}
    for subject in np.unique(subjects[test]):
        mask = subjects[test] == subject
        per_subject[str(subject)] = float(balanced_accuracy_score(y[test][mask],predicted[mask]))
    return dict(validation_balanced_accuracy=float(score), selected_C=c,
                test_balanced_accuracy=float(balanced_accuracy_score(y[test],predicted)),
                test_macro_f1=float(f1_score(y[test],predicted,average='macro')),
                test_auroc=float(roc_auc_score(y[test],probabilities)),
                confusion_matrix=confusion_matrix(y[test],predicted,labels=[0,1]).tolist(),
                per_subject_balanced_accuracy=per_subject)


def run_experiment(data, include_models=True, random_controls=True):
    X,y,subjects = data['X'],data['y'],data['subjects']
    masks, ids = split_subjects(subjects)
    results = {'split':dict(zip(['train','validation','test'],[p.tolist() for p in ids])),
               'note':'Frozen representations plus validation-selected logistic regression; not paper replication.',
               'pretraining_overlap':'LaBraM Appendix D includes this PhysioNet corpus. '
                    'Supervised subject isolation does not establish unseen-pretraining-data transfer.',
               'methods':{}}
    features = bandpower_features(X)
    results['methods']['bandpower'] = evaluate(features,y,subjects,masks)
    # A recording-level label-permutation sanity check respects window grouping.
    rng = np.random.default_rng(42)
    flipped = {s:bool(rng.integers(2)) for s in np.unique(subjects)}
    shuffled_y = np.array([int(v)^int(flipped[s]) for v,s in zip(y,subjects)])
    results['methods']['bandpower_label_control'] = evaluate(features,shuffled_y,subjects,masks)
    if include_models:
        for kind in ['cbramod','labram']:
            for pretrained in ([True,False] if random_controls else [True]):
                label = f'{kind}_' + ('pretrained' if pretrained else 'random')
                print('Extracting',label,flush=True)
                representation = extract_features(X,kind,pretrained)
                np.save(ROOT / f'{label}_features.npy',representation)
                results['methods'][label] = evaluate(representation,y,subjects,masks)
                (ROOT / 'results.json').write_text(json.dumps(results,indent=2))
    (ROOT / 'results.json').write_text(json.dumps(results,indent=2))
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['setup','prepare','baseline','all'])
    args = parser.parse_args()
    if args.action == 'setup':
        setup_assets()
    elif args.action == 'prepare':
        data = load_public_data()
        print('Saved',data['X'].shape,'windows/channels/samples to dataset.npz')
    else:
        with np.load(ROOT / 'dataset.npz',allow_pickle=False) as saved:
            data = dict(saved)
        print(json.dumps(run_experiment(data,include_models=args.action=='all'),indent=2))
