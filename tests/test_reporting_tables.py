"""Check method selection and differences on results with known answers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.table_functions import (
    build_best_methods_table, build_effect_table, build_gain_table,
)


class ReportingTableTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'L-SLR_AUC': [.9, .1, .2], 'QDA_AUC': [.65, .65, .65],
            'TabPFN_v25_AUC': [.6, .8, .7], 'TabPFN_v3_AUC': [.95, .4, .85],
            'L-SLR_Brier': [.2, .2, .2], 'QDA_Brier': [.4, .4, .4],
            'TabPFN_v25_Brier': [.1, .12, .14], 'TabPFN_v3_Brier': [.18, .2, .22],
            'L-SLR_CalSlope': [0., 2., 1.], 'QDA_CalSlope': [1.1, 1.1, 1.1],
            'TabPFN_v25_CalSlope': [.5, 1.5, 1.], 'TabPFN_v3_CalSlope': [.95, .95, .95],
            'L-SLR_CalIntercept': [-1., 1., 0.], 'QDA_CalIntercept': [.1, .1, .1],
            'TabPFN_v25_CalIntercept': [-.5, .5, 0.], 'TabPFN_v3_CalIntercept': [.05, .05, .05],
        })
        self.results = {'S1': self.frame}

    def test_best_two_are_ranked_within_each_group_and_measure(self):
        table = build_best_methods_table(self.results, metrics=['AUC', 'Brier'], filename=None).data
        row = table.loc[('S1', 'AUC', 'Classical')]
        self.assertEqual(row['Best method'], 'QDA')
        self.assertEqual(row['Second method'], 'L-SLR')
        self.assertAlmostEqual(row['Best mean'], .65)
        self.assertAlmostEqual(row['Second SD'], np.std([.9, .1, .2], ddof=1))
        self.assertEqual(table.loc[('S1', 'AUC', 'Foundation'), 'Best method'], 'TabPFN_v3')
        self.assertEqual(table.loc[('S1', 'Brier', 'Foundation'), 'Best method'], 'TabPFN_v25')
        self.assertEqual(table.loc[('S1', 'Brier', 'Classical'), 'Best method'], 'L-SLR')

    def test_effect_uses_fixed_winners_and_sd_of_paired_differences(self):
        table = build_effect_table(self.results, metrics=['AUC', 'Brier'], filename=None).data
        row = table.loc[('S1', 'AUC')]
        differences = np.array([.95, .4, .85]) - .65
        self.assertEqual(row['Classical method'], 'QDA')
        self.assertEqual(row['Foundation method'], 'TabPFN_v3')
        self.assertAlmostEqual(row['Mean effect'], differences.mean())
        self.assertAlmostEqual(row['SD effect'], differences.std(ddof=1))
        self.assertEqual(row['Paired runs'], 3)
        self.assertAlmostEqual(table.loc[('S1', 'Brier'), 'Mean effect'], -.08)

    def test_every_model_is_compared_with_best_in_other_group(self):
        table = build_effect_table(self.results, metrics=['AUC'], filename=None, all_models=True).data
        pairs = set(zip(table['Foundation method'], table['Classical method']))
        self.assertEqual(pairs, {
            ('TabPFN_v3', 'QDA'), ('TabPFN_v25', 'QDA'), ('TabPFN_v3', 'L-SLR'),
        })
        self.assertEqual(len(table), 3)

    def test_missing_values_keep_repetition_pairing_and_complete_pair_count(self):
        frame = pd.DataFrame({
            'QDA_AUC': [.6, np.nan, .8, .7],
            'TabICL-V1_AUC': [.8, .7, np.nan, .9],
        })
        row = build_effect_table({'D1': frame}, metrics=['AUC'], filename=None).data.iloc[0]
        self.assertEqual(row['Paired runs'], 2)
        self.assertAlmostEqual(row['Mean effect'], .2)
        self.assertAlmostEqual(row['SD effect'], 0)

    def test_calibration_ranks_absolute_deviations_without_cancellation(self):
        table = build_best_methods_table(
            self.results, metrics=['CalSlope', 'CalIntercept'], filename=None,
        ).data
        for metric in ('CalSlope', 'CalIntercept'):
            self.assertEqual(table.loc[('S1', metric, 'Classical'), 'Best method'], 'QDA')
            self.assertEqual(table.loc[('S1', metric, 'Foundation'), 'Best method'], 'TabPFN_v3')
        effect = build_effect_table(self.results, metrics=['CalSlope'], filename=None).data.iloc[0]
        self.assertAlmostEqual(effect['Mean effect'], -.15)

    def test_sample_size_changes_use_independent_means_with_different_run_counts(self):
        base = {'S1': pd.DataFrame({'QDA_AUC': [.5, .7], 'QDA_Brier': [.3, .4]})}
        sens = {
            'S1': pd.DataFrame({'QDA_AUC': [.8, .8, .8], 'QDA_Brier': [.1, .1, .1]}),
            'S7+': pd.DataFrame({'QDA_AUC': [.9]}),
        }
        with tempfile.TemporaryDirectory() as directory:
            table = build_gain_table(
                base, sens, metrics=['AUC', 'Brier'], methods=['QDA'],
                tables_dir=directory, filename='changes.csv',
            )
            self.assertAlmostEqual(table.data.loc[('S1', 'AUC'), 'QDA'], .2)
            self.assertAlmostEqual(table.data.loc[('S1', 'Brier'), 'QDA'], -.25)
            self.assertNotIn('S7+', table.data.index.get_level_values('Setting'))
            self.assertTrue((Path(directory) / 'changes.csv').is_file())
            table.to_html()
        legacy = build_gain_table(base, sens, methods=['QDA'], filename=None).data
        self.assertEqual(legacy.index.names, ['Setting'])
        self.assertAlmostEqual(legacy.loc['S1', 'QDA'], .2)

    def test_unavailable_models_are_not_selected_as_winners(self):
        frame = pd.DataFrame({'QDA_AUC': [.7, .8], 'TabICL-V1_AUC': [np.nan, np.nan]})
        summary = build_best_methods_table({'S1': frame}, metrics=['AUC'], filename=None).data
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['Second method'], '—')
        self.assertIsNone(build_effect_table({'S1': frame}, metrics=['AUC'], filename=None))


if __name__ == '__main__':
    unittest.main()
