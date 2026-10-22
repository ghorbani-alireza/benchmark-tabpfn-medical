"""Check QDA's holdout selection, scaling and high-dimensional predictions."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmark.model_definitions import run_qda
from benchmark.plot_functions import CLASSICAL_METHODS, METHOD_ORDER
from benchmark.table_functions import METHODS


class QDAGridSearchTests(unittest.TestCase):
    def test_validation_selection_scaling_and_single_final_refit(self):
        rng = np.random.default_rng(2025)
        y_train = np.repeat([0, 1], 40)
        X_train = rng.normal(size=(80, 4)) * [1., 2., 5., 10.] + y_train[:, None]
        y_test = np.repeat([0, 1], 10)
        X_test = rng.normal(size=(20, 4)) * [1., 2., 5., 10.] + y_test[:, None]
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train, y_train, test_size=.2, stratify=y_train, random_state=2025,
        )
        candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        tune_scaler = StandardScaler().fit(X_sub)
        X_sub_scaled = tune_scaler.transform(X_sub)
        X_val_scaled = tune_scaler.transform(X_val)
        scores = [roc_auc_score(y_val, QuadraticDiscriminantAnalysis(
            reg_param=value,
        ).fit(X_sub_scaled, y_sub).decision_function(X_val_scaled)) for value in candidates]
        best_reg_param = candidates[np.argmax(scores)]
        final_scaler = StandardScaler().fit(X_train)
        X_train_scaled = final_scaler.transform(X_train)
        X_test_scaled = final_scaler.transform(X_test)
        expected = QuadraticDiscriminantAnalysis(reg_param=best_reg_param).fit(
            X_train_scaled, y_train,
        )
        fits, probability_calls = [], []

        class RecordingQDA(QuadraticDiscriminantAnalysis):
            def fit(self, X, y):
                fits.append((X.copy(), y.copy(), self.reg_param))
                return super().fit(X, y)

            def predict_proba(self, X):
                probability_calls.append(X.copy())
                return super().predict_proba(X)

        with patch('sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis', RecordingQDA):
            result = run_qda(
                X_train, X_test, y_train, y_test, X_sub, y_sub, X_val, y_val,
                random_state=2025,
            )

        self.assertEqual(len(fits), len(candidates) + 1)
        for (features, labels, regularization), candidate in zip(fits[:-1], candidates):
            np.testing.assert_allclose(features, X_sub_scaled)
            np.testing.assert_array_equal(labels, y_sub)
            self.assertEqual(regularization, candidate)
        np.testing.assert_allclose(fits[-1][0], X_train_scaled)
        np.testing.assert_array_equal(fits[-1][1], y_train)
        self.assertEqual(fits[-1][2], best_reg_param)
        self.assertEqual(len(probability_calls), 1)
        np.testing.assert_allclose(probability_calls[0], X_test_scaled)
        expected_proba = expected.predict_proba(X_test_scaled)[:, 1]
        expected_pred = expected.predict(X_test_scaled)
        np.testing.assert_allclose(result[5], expected_proba)
        self.assertAlmostEqual(result[0], balanced_accuracy_score(y_test, expected_pred))
        self.assertAlmostEqual(result[1], roc_auc_score(y_test, expected_proba))
        self.assertAlmostEqual(result[2], np.mean((expected_pred - y_test) ** 2))
        self.assertAlmostEqual(result[6], np.mean((expected_proba - y_test) ** 2))

    def test_singular_covariance_and_constant_predictor_keep_finite_probabilities(self):
        rng = np.random.default_rng(8)
        y_train = np.repeat([0, 1], 100)
        X_train = rng.normal(size=(200, 100)) + .2 * y_train[:, None]
        X_train[:, -1] = 1.
        y_test = np.repeat([0, 1], 10)
        X_test = rng.normal(size=(20, 100)) + .2 * y_test[:, None]
        X_test[:, -1] = 1.
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train, y_train, test_size=.2, stratify=y_train, random_state=8,
        )
        # Rank warnings are expected when there are more features than class rows.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            result = run_qda(
                X_train, X_test, y_train, y_test, X_sub, y_sub, X_val, y_val,
                random_state=8,
            )
        self.assertEqual(len(result), 11)
        self.assertEqual(result[5].shape, y_test.shape)
        self.assertTrue(np.isfinite(result[5]).all())
        self.assertTrue(((result[5] >= 0) & (result[5] <= 1)).all())
        self.assertTrue(np.isfinite(result[1]))
        self.assertTrue(all(value >= 0 for value in result[7:]))

    def test_qda_precedes_quadro_in_reporting(self):
        for methods in (CLASSICAL_METHODS, METHOD_ORDER, METHODS):
            self.assertEqual(methods.index('QDA') + 1, methods.index('Quadro'))


if __name__ == '__main__':
    unittest.main()
