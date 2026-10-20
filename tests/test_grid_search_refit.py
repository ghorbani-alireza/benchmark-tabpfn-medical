"""Ensure tuned tree runners perform only one final full-training fit."""
import unittest
from unittest.mock import patch

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from benchmark.model_definitions import run_catboost, run_random_forest, run_xgboost


class GridSearchRefitTests(unittest.TestCase):
    def test_tree_runners_refit_once(self):
        rng = np.random.default_rng(12)
        y = np.tile([0, 1], 30)
        X = rng.normal(size=(60, 3)) + y[:, None]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, stratify=y, random_state=12,
        )
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train, y_train, test_size=.25, stratify=y_train, random_state=13,
        )
        for runner, model_class in (
            (run_random_forest, RandomForestClassifier),
            (run_xgboost, XGBClassifier),
            (run_catboost, CatBoostClassifier),
        ):
            with self.subTest(runner=runner.__name__):
                calls = []
                original_fit = model_class.fit

                def recording_fit(
                    model, X, y, *args, _calls=calls, _fit=original_fit, **kwargs,
                ):
                    _calls.append((X.copy(), y.copy()))
                    return _fit(model, X, y, *args, **kwargs)

                with patch.object(model_class, 'fit', recording_fit):
                    result = runner(
                        X_train, X_test, y_train, y_test,
                        X_sub, y_sub, X_val, y_val, random_state=12,
                    )

                self.assertEqual(len(calls), 10)  # Nine candidates plus one refit.
                for fit_X, fit_y in calls[:9]:
                    np.testing.assert_array_equal(fit_X, X_sub)
                    np.testing.assert_array_equal(fit_y, y_sub)
                np.testing.assert_array_equal(calls[-1][0], X_train)
                np.testing.assert_array_equal(calls[-1][1], y_train)
                self.assertTrue(np.isfinite(result[1]))


if __name__ == '__main__':
    unittest.main()
