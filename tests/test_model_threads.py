"""Check native prediction and optimization controls with small real models."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from catboost import CatBoostClassifier
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split

from benchmark.model_definitions import run_catboost
from quadro.moments import clime_precision


class ModelThreadTests(unittest.TestCase):
    def test_catboost_validation_and_test_prediction_threads(self):
        rng = np.random.default_rng(20)
        y = np.repeat([0, 1], 20)
        X = rng.normal(size=(40, 2)) + y[:, None]
        X_sub, X_val, y_sub, y_val = train_test_split(
            X, y, test_size=.25, stratify=y, random_state=20,
        )
        original = CatBoostClassifier._predict
        prediction_threads = []

        def checked_prediction(*args, **kwargs):
            parameters = inspect.signature(original).bind(*args, **kwargs).arguments
            prediction_threads.append(parameters['thread_count'])
            return original(*args, **kwargs)

        with patch.object(CatBoostClassifier, '_predict', checked_prediction):
            result = run_catboost(X, X_val, y, y_val, X_sub, y_sub, X_val, y_val,
                                 random_state=20)
        # Nine validation predictions, followed by test labels and probabilities.
        self.assertEqual(len(prediction_threads), 11)
        self.assertEqual(set(prediction_threads), {1})
        self.assertTrue(np.isfinite(result[1]))

    def test_quadro_highs_thread_option(self):
        with patch('scipy.optimize.linprog', wraps=linprog) as optimizer:
            precision = clime_precision(np.eye(2))
        self.assertEqual(optimizer.call_count, 2)
        for call in optimizer.call_args_list:
            self.assertEqual(call.kwargs['options']['threads'], 1)
        self.assertTrue(np.isfinite(precision).all())


if __name__ == '__main__':
    unittest.main()
