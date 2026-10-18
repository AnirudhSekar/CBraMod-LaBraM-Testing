"""Synthetic evaluation checks; these do not measure real EEG performance."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from eeg_cross_validation import evaluate_representations

class NestedCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(23)
        cls.subjects=np.repeat(np.arange(1,21),6)
        cls.y=np.tile([0,0,0,1,1,1],20)
        cls.strong=rng.normal(0,.1,(len(cls.y),5))
        cls.strong[:,0]+=3*cls.y
        cls.weak=rng.normal(0,1,(len(cls.y),5))
        with threadpool_limits(limits=1):
            cls.report=evaluate_representations(
                {'cbramod_pretrained':cls.strong,'cbramod_random':cls.weak},
                cls.y,cls.subjects,repeats=2,verbose=False)

    def test_subjects_isolated_and_each_tested_once_per_repeat(self):
        self.assertEqual(len(self.report['folds']),10)
        for fold in self.report['folds']:
            train,test=set(fold['train_subjects']),set(fold['test_subjects'])
            self.assertEqual((len(train),len(test)),(16,4))
            self.assertFalse(train&test)
            for inner in fold['inner_subject_splits']:
                t,v=set(inner['train']),set(inner['validation'])
                self.assertEqual((len(t),len(v)),(12,4))
                self.assertFalse(t&v)
                self.assertEqual(t|v,train)
                self.assertFalse((t|v)&test)
        for rep in [1,2]:
            rows=[r for r in self.report['participant_scores']
                  if r['repeat']==rep and r['method']=='cbramod_pretrained']
            self.assertEqual(len(rows),20)
            self.assertEqual(len({r['subject'] for r in rows}),20)

    def test_known_signal_and_equal_participant_aggregation(self):
        score=self.report['summary']['cbramod_pretrained']['balanced_accuracy']['mean']
        self.assertGreater(score,.95)
        rows=[r['balanced_accuracy'] for r in self.report['participant_scores']
              if r['method']=='cbramod_pretrained']
        self.assertAlmostEqual(score,float(np.mean(rows)))
        self.assertGreater(self.report['paired_pretraining_differences']['cbramod']['mean_percentage_points'],20)

    def test_output_predictions_cover_every_window(self):
        with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=1):
            evaluate_representations({'signal':self.strong},self.y,self.subjects,
                                     repeats=1,output_dir=temp,verbose=False)
            with np.load(Path(temp)/'cv_oof_predictions.npz') as saved:
                self.assertEqual(saved['signal_predictions'].shape,(1,len(self.y)))
                self.assertTrue(np.isfinite(saved['signal_probabilities']).all())
            for name in ['cv_results.json','cv_summary.csv','cv_participants.csv']:
                self.assertTrue((Path(temp)/name).exists())

    def test_invalid_features_rejected(self):
        bad=self.strong.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):
            evaluate_representations({'bad':bad},self.y,self.subjects,verbose=False)

if __name__=='__main__':unittest.main()
