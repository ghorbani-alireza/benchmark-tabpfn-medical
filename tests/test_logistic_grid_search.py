"""Verify holdout scaling, penalty selection, and the single final refit."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from benchmark.model_definitions import (
    run_linear_sparse_logistic_regression,
    run_sparse_logistic_regression,
)


class LogisticGridSearchTests(unittest.TestCase):
    def check_runner(self, runner, quadratic):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(120, 3))
        y = (X[:, 0] + .6 * X[:, 1] + rng.normal(size=120) > 0).astype(int)
        # Different measurement units expose missing or inconsistent scaling.
        X = X * [1., 1000., .01] + [3., 100., .1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25, stratify=y, random_state=12,
        )
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train, y_train, test_size=.3, stratify=y_train, random_state=13,
        )
        if quadratic:
            # Independent construction of the same linear and quadratic terms.
            expand = PolynomialFeatures(degree=2, include_bias=False)
            train_features = expand.fit_transform(X_sub)
            val_features = expand.transform(X_val)
            test_features = expand.transform(X_test)
            full_features = expand.transform(X_train)
        else:
            train_features, val_features, test_features = X_sub, X_val, X_test
            full_features = X_train
        full_y = y_train

        # Reference: tune using only the intended holdout, then fit once on all rows.
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        val_scaled = scaler.transform(val_features)
        scores = []
        penalties = np.logspace(-4, 4, 10)
        for C in penalties:
            model = LogisticRegression(
                penalty='l1', solver='liblinear', C=C, max_iter=2000,
                n_jobs=1, random_state=12,
            ).fit(train_scaled, y_sub)
            scores.append(roc_auc_score(y_val, model.predict_proba(val_scaled)[:, 1]))
        best_C = penalties[np.argmax(scores)]
        final_scaler = StandardScaler().fit(full_features)
        reference = LogisticRegression(
            penalty='l1', solver='liblinear', C=best_C, max_iter=2000,
            n_jobs=1, random_state=12,
        ).fit(final_scaler.transform(full_features), full_y)
        expected_proba = reference.predict_proba(final_scaler.transform(test_features))[:, 1]
        expected_pred = reference.predict(final_scaler.transform(test_features))

        scaler_inputs, model_inputs = [], []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                scaler_inputs.append(X.copy())
                return super().fit(X, y, sample_weight=sample_weight)

        class RecordingLogisticRegression(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                model_inputs.append((X.copy(), y.copy(), self.C))
                return super().fit(X, y, sample_weight=sample_weight)

        with patch('benchmark.model_definitions.StandardScaler', RecordingScaler), patch(
            'benchmark.model_definitions.LogisticRegression', RecordingLogisticRegression,
        ):
            result = runner(
                X_train, X_test, y_train, y_test, X_sub, y_sub, X_val, y_val,
                random_state=12,
            )

        self.assertEqual(len(scaler_inputs), 2)
        self.assertEqual(len(model_inputs), 11)
        np.testing.assert_allclose(scaler_inputs[0], train_features)
        for scaled, labels, _ in model_inputs[:10]:
            np.testing.assert_allclose(scaled, train_scaled)
            np.testing.assert_array_equal(labels, y_sub)
        np.testing.assert_allclose(scaler_inputs[-1], full_features)
        np.testing.assert_allclose(model_inputs[-1][0], final_scaler.transform(full_features))
        np.testing.assert_array_equal(model_inputs[-1][1], full_y)
        self.assertEqual(model_inputs[-1][2], best_C)
        np.testing.assert_allclose(result[5], expected_proba)
        self.assertAlmostEqual(result[0], balanced_accuracy_score(y_test, expected_pred))
        self.assertAlmostEqual(result[1], roc_auc_score(y_test, expected_proba))
        self.assertAlmostEqual(result[2], np.mean((expected_pred - y_test) ** 2))
        self.assertAlmostEqual(result[6], np.mean((expected_proba - y_test) ** 2))
        self.assertEqual(len(result), 11)
        self.assertTrue(all(value >= 0 for value in result[7:]))

    def test_linear_scaling_selection_and_refit(self):
        self.check_runner(run_linear_sparse_logistic_regression, quadratic=False)

    def test_quadratic_scaling_selection_and_refit(self):
        self.check_runner(run_sparse_logistic_regression, quadratic=True)


if __name__ == '__main__':
    unittest.main()
