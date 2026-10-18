"""Offline checks only. Synthetic scores are never experimental results."""
import numpy as np
from eeg_lab import bandpower_features, patchify, split_subjects, evaluate

def main():
    rng = np.random.default_rng(5)
    n_subjects, per_class = 20, 6
    subjects = np.repeat(np.arange(1,n_subjects+1),2*per_class)
    y = np.tile(np.repeat([0,1],per_class),n_subjects)
    X = rng.normal(0,.1,(len(y),19,800)).astype('float32')
    # Artificial, planted 10 Hz signal: verify frequency extraction and evaluation.
    wave = np.sin(2*np.pi*10*np.arange(800)/200)
    X[y==1,-2:,:] += wave
    features = bandpower_features(X)
    assert features.shape == (len(y),95)
    assert np.isfinite(features).all()
    patched = patchify(X)
    assert patched.shape == (len(y),19,4,200)
    np.testing.assert_array_equal(patched.reshape(X.shape),X)
    masks,ids = split_subjects(subjects)
    assert list(map(len,ids)) == [12,4,4]
    assert np.all(sum(m.astype(int) for m in masks)==1)
    assert all(len(np.unique(y[m]))==2 for m in masks)
    result = evaluate(features,y,subjects,masks)
    assert result['test_balanced_accuracy'] > .95
    assert sum(map(sum,result['confusion_matrix'])) == sum(masks[2])
    # Feature standardization/selection must ignore held-out data during fitting.
    disturbed = features.copy()
    disturbed[masks[2]] *= 1e6
    second = evaluate(disturbed,y,subjects,masks)
    assert result['selected_C'] == second['selected_C']
    assert result['validation_balanced_accuracy'] == second['validation_balanced_accuracy']
    try:
        patchify(X[...,:799])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid input shape accepted')
    print('PASS: shapes, patch round trip, bandpowers, subject isolation, validation/test separation.')
    print('Synthetic input only. No real-data accuracy or model execution has been measured.')

if __name__ == '__main__':
    main()
