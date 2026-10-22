"""Check single-pass probabilities and preservation of native class decisions."""
import sys
import types
import unittest
from functools import wraps
from unittest.mock import patch

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from benchmark import model_definitions as models


class PredictionReuseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(36)
        y = np.tile([0, 1], 40)
        X = rng.normal(size=(80, 3)) + y[:, None]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=.25, stratify=y, random_state=36,
        )
        self.split = train_test_split(
            self.X_train, self.y_train, test_size=.3,
            stratify=self.y_train, random_state=37,
        )

    def test_classical_runners_match_native_labels_without_predict_calls(self):
        X_sub, X_val, y_sub, y_val = self.split
        for runner, classifier in (
            (models.run_linear_sparse_logistic_regression, LogisticRegression),
            (models.run_sparse_logistic_regression, LogisticRegression),
            (models.run_random_forest, RandomForestClassifier),
            (models.run_xgboost, XGBClassifier),
            (models.run_catboost, CatBoostClassifier),
            (models.run_qda, QuadraticDiscriminantAnalysis),
        ):
            with self.subTest(runner=runner.__name__):
                calls = []
                original_proba = classifier.predict_proba

                @wraps(original_proba)
                def record_proba(model, X, *args, _calls=calls, _proba=original_proba, **kwargs):
                    probabilities = _proba(model, X, *args, **kwargs)
                    _calls.append((model, X.copy(), probabilities.copy()))
                    return probabilities

                with patch.object(classifier, 'predict_proba', record_proba), patch.object(
                    classifier, 'predict', side_effect=AssertionError('Repeated prediction'),
                ), patch.object(
                    models, 'balanced_accuracy_score', wraps=balanced_accuracy_score,
                ) as label_metric:
                    result = runner(
                        self.X_train, self.X_test, self.y_train, self.y_test,
                        X_sub, y_sub, X_val, y_val, random_state=36,
                    )

                # Logistic and QDA tuning use decision scores; trees use probabilities.
                self.assertEqual(len(calls), 1 if classifier in (LogisticRegression, QuadraticDiscriminantAnalysis) else 10)
                final_model, test_features, probabilities = calls[-1]
                kwargs = {'thread_count': 1} if classifier is CatBoostClassifier else {}
                native_pred = final_model.predict(test_features, **kwargs)
                np.testing.assert_array_equal(label_metric.call_args.args[1], native_pred)
                np.testing.assert_array_equal(result[5], probabilities[:, 1])
                self.assertAlmostEqual(result[0], balanced_accuracy_score(self.y_test, native_pred))
                self.assertAlmostEqual(result[2], np.mean((native_pred - self.y_test) ** 2))
                self.assertEqual(len(result), 11)
                self.assertTrue(all(value >= 0 for value in result[7:]))

    def test_foundation_runners_use_one_probability_call(self):
        probabilities = np.array([[.8, .2], [.5, .5], [.1, .9], [.7, .3]])
        X_test = self.X_test[:4]
        y_test = np.array([0, 0, 1, 1])
        for runner, package, class_name in (
            (models.run_tabpfn_v25, 'tabpfn_v25', 'TabPFNClassifier'),
            (models.run_tabpfn_v26, 'tabpfn_v26', 'TabPFNClassifier'),
            (models.run_tabpfn_v3, 'tabpfn_v3', 'TabPFNClassifier'),
            (models.run_tabpfn_v35, 'tabpfn_v35', 'TabPFNClassifier'),
            (models.run_exaone, 'exaonetabular', 'EXAONETabularClassifier'),
            (models.run_tabicl_v1, 'tabicl', 'TabICLClassifier'),
            (models.run_tabicl, 'tabicl', 'TabICLClassifier'),
        ):
            with self.subTest(runner=runner.__name__):
                calls = []

                class FakeClassifier:
                    def __init__(self, **kwargs):
                        pass

                    @classmethod
                    def from_pretrained(cls, **kwargs):
                        return cls(**kwargs)

                    def fit(self, X, y):
                        self.classes_ = np.unique(y)
                        return self

                    def predict(self, X):
                        raise AssertionError('Repeated prediction')

                    def predict_proba(self, X, _calls=calls):
                        _calls.append(X.copy())
                        return probabilities.copy()

                module = types.ModuleType(package)
                setattr(module, class_name, FakeClassifier)
                with patch.dict(sys.modules, {package: module}), patch.object(
                    models, '_configure_pretrained_credentials',
                ):
                    result = runner(
                        self.X_train, X_test, self.y_train, y_test,
                        random_state=36,
                    )

                self.assertEqual(len(calls), 1)
                np.testing.assert_array_equal(calls[0], X_test)
                np.testing.assert_array_equal(result[5], probabilities[:, 1])
                # A .5/.5 tie selects the first class, matching native argmax.
                expected_pred = np.array([0, 0, 1, 0])
                self.assertAlmostEqual(result[0], balanced_accuracy_score(y_test, expected_pred))
                self.assertAlmostEqual(result[2], np.mean((expected_pred - y_test) ** 2))
                self.assertEqual(len(result), 11)
                self.assertEqual(result[8], 0.0)
                self.assertTrue(all(value >= 0 for value in result[7:]))


if __name__ == '__main__':
    unittest.main()
