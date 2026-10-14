"""Check the distribution-based reference without fitting benchmark models."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.integrate import quad

from benchmark.settings import calculate_bayes_auc
from benchmark.utils import read_bayes_auc
from benchmark.plot_functions import plot_boxplots_grid, plot_boxplots_grid_lands


class BayesAUCTests(unittest.TestCase):
    def setting(self, dist='Normal', variance=4):
        # Zero means isolate the information carried by different variances.
        return dict(mu={'mu1': [0.0], 'mu2': [0.0]}, sigma1=[[1.0]],
                    sigma2=[[float(variance)]], dist=dist, df=5)

    def test_exact_equal_covariance(self):
        cfg = self.setting(variance=1)
        cfg['mu']['mu2'] = [1.0]
        result = calculate_bayes_auc(cfg)
        self.assertAlmostEqual(result['auc'], norm.cdf(1 / np.sqrt(2)))
        self.assertEqual(result['mc_se'], 0)

    def test_unequal_normal_against_exact_probability(self):
        # For zero means the likelihood ratio ranks |x|. Independent normal
        # ratios have a Cauchy distribution, giving this exact probability.
        expected = 2 / np.pi * np.arctan(2)
        result = calculate_bayes_auc(self.setting(), n_pairs=100_000)
        self.assertLess(abs(result['auc'] - expected), 4 * result['mc_se'])
        self.assertGreater(result['mc_se'], 0)

    def test_t_against_independent_integration(self):
        # For t classes sharing df and mean, the score also ranks |x|.
        # Common covariance-to-shape factors cancel in this probability.
        expected = quad(lambda x: 2 * t.pdf(x, 5) * (2 * t.cdf(2*x, 5) - 1), 0, np.inf)[0]
        result = calculate_bayes_auc(self.setting('t'))
        self.assertLess(abs(result['auc'] - expected), 4 * result['mc_se'])

    def test_identical_t_classes_ties(self):
        result = calculate_bayes_auc(self.setting('t', variance=1), n_pairs=100)
        self.assertEqual(result['auc'], 0.5)
        self.assertEqual(result['mc_se'], 0)

    def test_reproducible_without_changing_simulation_rng(self):
        np.random.seed(12)
        state = np.random.get_state()
        first = calculate_bayes_auc(self.setting(), n_pairs=123)
        second = calculate_bayes_auc(self.setting(), n_pairs=123)
        self.assertEqual(first, second)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_metadata_loading_and_missing_distributions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sim.csv'
            meta = path.with_suffix('.meta.json')
            saved = calculate_bayes_auc(self.setting(), n_pairs=100)
            meta.write_text(json.dumps({'bayes_auc': saved}))
            self.assertEqual(read_bayes_auc({'S1': path})['S1'], saved)
            meta.write_text(json.dumps(self.setting(variance=1)))
            self.assertEqual(read_bayes_auc({'S1': path})['S1']['auc'], 0.5)
            meta.write_text(json.dumps({'dataset': 'real'}))
            self.assertEqual(read_bayes_auc({'D1': path}), {})

    def test_plot_toggle_and_auc_only(self):
        results = {'S1': pd.DataFrame({'SLR_AUC': [.6, .7, .8], 'SLR_MSE': [.2, .3, .4]})}
        reference = {'S1': {'auc': .9, 'mc_se': .001}}
        for plot in (plot_boxplots_grid, plot_boxplots_grid_lands):
            for metric, enabled, expected in [('AUC', True, 1), ('AUC', False, 0), ('MSE', True, 0)]:
                plot(results, ['S1'], metric=metric, bayes_auc=reference, show_bayes_auc=enabled)
                ax = plt.gcf().axes[0]
                lines = [line for line in ax.lines if line.get_label().startswith('Bayes AUC')]
                self.assertEqual(len(lines), expected)
                if lines:
                    np.testing.assert_array_equal(lines[0].get_ydata(), [.9, .9])
                plt.close('all')


if __name__ == '__main__':
    unittest.main()
