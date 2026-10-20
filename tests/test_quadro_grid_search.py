"""Check QUADRO's grid search against the previous validation procedure."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from benchmark.model_definitions import run_quadro
from quadro import QUADROClassifier


class QuadroGridSearchTests(unittest.TestCase):
    def test_same_candidates_split_and_predictions(self):
        """Preserve selection and test predictions, with exactly one final refit."""
        rng = np.random.default_rng(2025)
        y_train = np.repeat([0, 1], 40)
        X_train = rng.normal(size=(80, 2)) + y_train[:, None]
        y_test = np.repeat([0, 1], 10)
        X_test = rng.normal(size=(20, 2)) + y_test[:, None]
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train, y_train, test_size=.2, stratify=y_train, random_state=2025,
        )
        pairs = [(0., 0.), (.01, .01), (.01, .1), (.1, .1), (.1, 1.)]
        X_sub_scaled = StandardScaler().fit_transform(X_sub)
        X_train_scaled = StandardScaler().fit_transform(X_train)

        # Reproduce the original manual search as a regression reference.
        best_auc = -np.inf
        with threadpool_limits(limits=1):
            for quadratic, linear in pairs:
                candidate = QUADROClassifier(lambda_quadratic=quadratic, lambda_linear=linear)
                candidate.fit(X_sub, y_sub)
                score = roc_auc_score(y_val, candidate.predict_proba(X_val)[:, 1])
                if score > best_auc:
                    best_auc, best_pair = score, (quadratic, linear)
            original = QUADROClassifier(
                lambda_quadratic=best_pair[0], lambda_linear=best_pair[1],
            ).fit(X_train, y_train)
            expected_proba = original.predict_proba(X_test)[:, 1]
            expected_pred = original.predict(X_test)

        calls = []

        class RecordingQUADRO(QUADROClassifier):
            def fit(self, X, y):
                calls.append((X.copy(), y.copy(), self.lambda_quadratic, self.lambda_linear))
                return super().fit(X, y)

        with patch('quadro.QUADROClassifier', RecordingQUADRO), patch(
            'benchmark.model_definitions.GridSearchCV', wraps=GridSearchCV,
        ) as search:
            result = run_quadro(
                X_train, X_test, y_train, y_test, X_sub, y_sub, X_val, y_val,
                random_state=2025,
            )

        self.assertEqual(len(calls), 6)  # Five validation fits, one final refit.
        for call, pair in zip(calls[:5], pairs):
            np.testing.assert_allclose(call[0], X_sub_scaled)
            np.testing.assert_array_equal(call[1], y_sub)
            self.assertEqual(call[2:], pair)
        np.testing.assert_allclose(calls[-1][0], X_train_scaled)
        np.testing.assert_array_equal(calls[-1][1], y_train)
        self.assertEqual(calls[-1][2:], best_pair)
        options = search.call_args.kwargs
        self.assertEqual(options['n_jobs'], 1)
        self.assertIs(options['refit'], False)
        self.assertIs(search.call_args.args[0].standardize, False)
        np.testing.assert_allclose(result[5], expected_proba, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(result[1], roc_auc_score(y_test, expected_proba))
        self.assertAlmostEqual(result[2], np.mean((expected_pred - y_test) ** 2))


if __name__ == '__main__':
    unittest.main()
