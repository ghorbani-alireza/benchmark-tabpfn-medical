"""Check original TabICL checkpoint selection and installed ensemble defaults."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
from tabicl import TabICLClassifier

from benchmark.model_definitions import run_tabicl_v1, run_tabicl
from benchmark.plot_functions import COLOR_SCHEME, FOUNDATION_METHODS, LINE_STYLES


class TabICLVersionTests(unittest.TestCase):
    def test_checkpoint_selection_uses_installed_default_ensemble(self):
        default = inspect.signature(TabICLClassifier).parameters['n_estimators'].default
        original_init = TabICLClassifier.__init__
        X = np.array([[0., 0.], [1., 1.], [2., 1.], [1., 0.]])
        y = np.array([0, 1, 1, 0])
        probabilities = np.array([[.8, .2], [.1, .9], [.2, .8], [.9, .1]])
        for runner, checkpoint in [
            (run_tabicl_v1, 'tabicl-classifier-v1-20250208.ckpt'),
            (run_tabicl, 'tabicl-classifier-v2-20260212.ckpt'),
        ]:
            with self.subTest(checkpoint=checkpoint):
                created = []

                def record_init(model, *args, **kwargs):
                    original_init(model, *args, **kwargs)
                    created.append((model, kwargs))

                def fit(model, train, outcomes):
                    model.classes_ = np.unique(outcomes)
                    return model

                with patch.object(TabICLClassifier, '__init__', record_init), patch.object(
                    TabICLClassifier, 'fit', fit,
                ), patch.object(TabICLClassifier, 'predict_proba', return_value=probabilities) as predict:
                    result = runner(X, X, y, y, random_state=41)
                model, options = created[0]
                self.assertNotIn('n_estimators', options)
                self.assertEqual(model.n_estimators, default)
                self.assertEqual(model.checkpoint_version, checkpoint)
                self.assertEqual(model.device, 'cpu')
                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(model.random_state, 41)
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(len(result), 11)
                np.testing.assert_array_equal(result[5], probabilities[:, 1])

    def test_versions_are_adjacent_with_distinct_colors_and_line_styles(self):
        self.assertEqual(FOUNDATION_METHODS.index('TabICL-V1') + 1, FOUNDATION_METHODS.index('TabICL-V2'))
        self.assertNotEqual(COLOR_SCHEME['TabICL-V1'], COLOR_SCHEME['TabICL-V2'])
        self.assertNotEqual(LINE_STYLES['TabICL-V1'], LINE_STYLES['TabICL-V2'])


if __name__ == '__main__':
    unittest.main()
