"""Tests de src/transformation.py. Lancer depuis la racine du projet :

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transformation import (fit_feature_params, group_rare_categories,  # noqa: E402
                            prepare_features, scale_minmax, scale_zscore)


def make_products():
    return pd.DataFrame({
        'id_produit': ['P1', 'P2', 'P3', 'P4'],
        'libelle': ['a', 'b', 'c', 'd'],
        'categorie': ['Audio', 'Audio', 'Mobilier', 'Informatique'],
        'sous_categorie': ['Casque', 'Enceinte', 'Chaise', 'Laptop'],
        'cout_achat': [50.0, 80.0, 40.0, 500.0],
    })


def make_sales(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'id_commande': [f'CMD{i}' for i in range(n)],
        'id_client': rng.choice([f'C{i:03d}' for i in range(30)], n),
        'id_produit': rng.choice(['P1', 'P2', 'P3', 'P4'], n),
        'quantite': rng.integers(1, 5, n),
        'prix_unitaire': rng.uniform(60, 1000, n).round(2),
        'remise_pct': rng.choice([0.0, 5.0, 10.0, 20.0], n),
        'canal': rng.choice(['web', 'boutique', 'telephone'], n),
        'statut': rng.choice(['livre', 'annule', 'retourne'], n),
    }).assign(montant=lambda d: d['quantite'] * d['prix_unitaire'] * (1 - d['remise_pct'] / 100))


class TestScaling(unittest.TestCase):
    def setUp(self):
        self.x = pd.Series([1.0, 2.0, 3.0, 10.0])

    def test_minmax_matches_formula(self):
        expected = (self.x - self.x.min()) / (self.x.max() - self.x.min())
        pd.testing.assert_series_equal(scale_minmax(self.x), expected)
        self.assertEqual(scale_minmax(self.x).min(), 0)
        self.assertEqual(scale_minmax(self.x).max(), 1)

    def test_zscore_matches_formula(self):
        expected = (self.x - self.x.mean()) / self.x.std()
        pd.testing.assert_series_equal(scale_zscore(self.x), expected)
        self.assertAlmostEqual(scale_zscore(self.x).mean(), 0)
        self.assertAlmostEqual(scale_zscore(self.x).std(), 1)

    def test_constant_series_gives_zeros_not_inf(self):
        constant = pd.Series([5.0, 5.0, 5.0])
        self.assertTrue((scale_minmax(constant) == 0).all())
        self.assertTrue((scale_zscore(constant) == 0).all())

    def test_learned_parameters_are_used_on_other_data(self):
        train, test = self.x.iloc[:3], self.x.iloc[3:]
        result = scale_zscore(test, train.mean(), train.std())
        self.assertAlmostEqual(result.iloc[0], (10.0 - train.mean()) / train.std())
        self.assertNotAlmostEqual(result.mean(), 0, msg='un test standardisé avec les paramètres du train n\'est pas centré')

    def test_missing_values_stay_missing(self):
        self.assertTrue(np.isnan(scale_zscore(pd.Series([1.0, np.nan, 3.0])).iloc[1]))


class TestGroupRare(unittest.TestCase):
    def test_rare_grouped_and_nan_kept(self):
        series = pd.Series(['a'] * 50 + ['b'] * 49 + ['rare'] + [None])
        result = group_rare_categories(series, min_frequency=0.05)
        self.assertEqual(set(result.dropna().unique()), {'a', 'b', 'autre'})
        self.assertTrue(result.isna().iloc[-1])


class TestPrepareFeatures(unittest.TestCase):
    def setUp(self):
        self.sales, self.products = make_sales(), make_products()
        self.features = prepare_features(self.sales, self.products)

    def test_does_not_modify_inputs(self):
        pd.testing.assert_frame_equal(self.sales, make_sales())
        pd.testing.assert_frame_equal(self.products, make_products())

    def test_row_count_is_preserved(self):
        self.assertEqual(len(self.features), len(self.sales))

    def test_net_amount_treats_discount_as_percent(self):
        row = self.sales.iloc[0]
        expected = row['quantite'] * row['prix_unitaire'] * (1 - row['remise_pct'] / 100)
        self.assertAlmostEqual(self.features['montant_net'].iloc[0], expected)
        self.assertTrue((self.features['montant_net'] > 0).all(), 'un montant net ne peut pas être négatif')

    def test_failed_order_flag_uses_real_statuses(self):
        flag = self.features['commande_non_aboutie']
        self.assertEqual(int(flag.sum()), int(self.sales['statut'].isin(['annule', 'retourne']).sum()))
        self.assertGreater(flag.sum(), 0, 'l\'indicateur ne doit pas être nul partout')

    def test_median_gap_is_centered_on_category(self):
        gaps = self.features.assign(cat=self.sales.merge(self.products, on='id_produit')['categorie'])
        medians = gaps.groupby('cat')['ecart_prix_median_cat'].median()
        self.assertTrue((medians.abs() < 1e-9).all())

    def test_client_average_matches_groupby(self):
        expected = self.sales.groupby('id_client')['montant'].transform('mean')
        np.testing.assert_allclose(self.features['panier_moyen_client'], expected)

    def test_margin_columns_and_no_infinite_values(self):
        self.assertIn('taux_marge', self.features.columns)
        numeric = self.features.select_dtypes('number')
        self.assertFalse(np.isinf(numeric).any().any())

    def test_zero_price_does_not_create_infinity(self):
        sales = self.sales.copy()
        sales.loc[0, 'prix_unitaire'] = 0.0
        features = prepare_features(sales, self.products)
        self.assertFalse(np.isinf(features['taux_marge']).any())

    def test_one_hot_columns_are_binary_and_replace_originals(self):
        for original in ['canal', 'statut', 'categorie', 'sous_categorie']:
            self.assertNotIn(original, self.features.columns)
        dummies = [c for c in self.features.columns if c.startswith(('canal=', 'statut=', 'cat=', 'souscat='))]
        self.assertGreater(len(dummies), 0)
        self.assertTrue(self.features[dummies].isin([0, 1]).all().all())
        self.assertTrue((self.features[[c for c in dummies if c.startswith('canal=')]].sum(axis=1) == 1).all())

    def test_zscore_column_is_standardized(self):
        self.assertAlmostEqual(self.features['montant_zscore'].mean(), 0)
        self.assertAlmostEqual(self.features['montant_zscore'].std(), 1)

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            prepare_features(self.sales.drop(columns=['remise_pct']), self.products)


class TestNoLeakage(unittest.TestCase):
    def setUp(self):
        sales, self.products = make_sales(), make_products()
        self.train, self.test = sales.iloc[:150], sales.iloc[150:]
        self.params = fit_feature_params(self.train, self.products)

    def test_test_set_is_scaled_with_train_parameters(self):
        features = prepare_features(self.test, self.products, params=self.params)
        expected = (self.test['montant'] - self.train['montant'].mean()) / self.train['montant'].std()
        np.testing.assert_allclose(features['montant_zscore'], expected)
        self.assertNotAlmostEqual(features['montant_zscore'].mean(), 0, places=6)

    def test_train_and_test_have_identical_columns(self):
        train_features = prepare_features(self.train, self.products, params=self.params)
        test_only_one_channel = self.test[self.test['canal'] == 'web']
        test_features = prepare_features(test_only_one_channel, self.products, params=self.params)
        self.assertEqual(list(train_features.columns), list(test_features.columns))

    def test_unseen_category_and_client_do_not_crash(self):
        new_products = pd.concat([self.products, pd.DataFrame({
            'id_produit': ['P9'], 'libelle': ['z'], 'categorie': ['Jardin'],
            'sous_categorie': ['Tondeuse'], 'cout_achat': [100.0]})], ignore_index=True)
        new_sale = self.test.iloc[[0]].assign(id_produit='P9', id_client='C999')
        features = prepare_features(new_sale, new_products, params=self.params)
        self.assertEqual(len(features), 1)
        self.assertTrue(np.isnan(features['panier_moyen_client'].iloc[0]))
        cat_columns = [c for c in features.columns if c.startswith('cat=')]
        self.assertEqual(int(features[cat_columns].sum().sum()), 0)

    def test_params_come_from_train_only(self):
        self.assertAlmostEqual(self.params['montant_mean'], self.train['montant'].mean())
        self.assertNotAlmostEqual(self.params['montant_mean'],
                                  pd.concat([self.train, self.test])['montant'].mean())


if __name__ == '__main__':
    unittest.main()
