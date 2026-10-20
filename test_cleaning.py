"""Tests de src/cleaning.py. Lancer depuis la racine du projet :

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cleaning import clean_sales, normalize_text, parse_mixed_dates, parse_price  # noqa: E402


def make_raw_sales():
    """Petit jeu dégradé qui reproduit les défauts de ventes_brutes.csv."""
    return pd.DataFrame({
        'id_commande': ['CMD1', 'CMD2', 'CMD3', 'CMD4', 'CMD5', 'CMD6', 'CMD7', 'CMD7', 'CMD8', 'CMD9'],
        'date_commande': ['2024/01/18 19:53', '08/01/2024', '2023-04-15', '06 Nov 2024',
                          '24-05-2023', None, '2023-07-29', '2023-07-29', 'pas une date', '2023-02-01'],
        'id_client': ['C02300', 'C02791', '990031', 'C01883', 'C00571', 'C00001', 'C01517', 'C01517', 'C00002', 'C00003'],
        'id_produit': ['P0009'] * 10,
        'id_magasin': ['M05'] * 10,
        'quantite': [3, 2, 3, 4, 2, 1, 2, 2, 1, -3],
        'prix_unitaire': ['160,66 EUR', 70.48, '691.12', '570,25 EUR', '1058,05 EUR',
                          '10,00 EUR', 100.0, 100.0, '5,00 EUR', '20,00 EUR'],
        'remise_pct': [5.0, np.nan, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'canal': ['web', 'WEB', '  web', 'Boutique', 'telephone', 'web', 'web', 'web', 'web', 'web'],
        'statut': ['livre', 'LIVRE', 'annule', 'livre', 'retourne', 'livre', 'livre', 'livre', 'livre', 'livre'],
        'ville_livraison': ['Lyon', 'LYON', 'lyon', 'Orléans', 'Paris', 'Paris', 'Paris', 'Paris', 'Paris', 'Paris'],
    })


class TestHelpers(unittest.TestCase):
    def test_parse_price_handles_comma_and_currency(self):
        result = parse_price(pd.Series(['129,12 EUR', '70.5', 60.9, 'abc']))
        self.assertEqual(result.iloc[:3].tolist(), [129.12, 70.5, 60.9])
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_parse_mixed_dates_reads_every_known_format(self):
        dates = pd.Series(['2024/01/18 19:53', '08/01/2024', '2023-04-15', '06 Nov 2024', '24-05-2023'])
        result = parse_mixed_dates(dates)
        self.assertEqual(result.notna().sum(), 5, 'aucune ligne ne doit être perdue à cause du format')
        self.assertEqual(result.iloc[1], pd.Timestamp('2024-01-08'), 'jour d\'abord pour jj/mm/aaaa')
        self.assertEqual(result.iloc[4], pd.Timestamp('2023-05-24'))

    def test_parse_mixed_dates_leaves_garbage_as_nat(self):
        result = parse_mixed_dates(pd.Series(['pas une date', None]))
        self.assertTrue(result.isna().all())

    def test_normalize_text_keeps_missing_values_missing(self):
        result = normalize_text(pd.Series(['  Orléans ', None, 'PARIS']))
        self.assertEqual(result.iloc[0], 'orleans')
        self.assertTrue(pd.isna(result.iloc[1]), "une valeur manquante ne doit pas devenir la chaîne 'nan'")
        self.assertEqual(result.iloc[2], 'paris')


class TestCleanSales(unittest.TestCase):
    def setUp(self):
        self.raw = make_raw_sales()
        self.clean, self.log = clean_sales(self.raw, verbose=False)

    def test_does_not_modify_input(self):
        original = make_raw_sales()
        pd.testing.assert_frame_equal(self.raw, original)

    def test_prices_are_parsed_not_destroyed(self):
        self.assertEqual(self.log['prix_unitaire_illisibles'], 0)
        self.assertEqual(self.log['montants_non_calculables'], 0)
        self.assertTrue(pd.api.types.is_float_dtype(self.clean['prix_unitaire']))

    def test_mixed_date_formats_do_not_cost_rows(self):
        # 10 lignes : -1 doublon, -1 date absente, -1 date illisible, -1 quantité négative
        self.assertEqual(self.log['lignes_finales'], 6)
        self.assertEqual(self.log['dates_manquantes_supprimees'], 1)
        self.assertEqual(self.log['dates_illisibles_supprimees'], 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.clean['date_commande']))

    def test_duplicates_and_negative_quantities_are_removed(self):
        self.assertEqual(self.log['doublons_supprimes'], 1)
        self.assertEqual(self.log['quantites_negatives_supprimees'], 1)
        self.assertTrue((self.clean['quantite'] >= 0).all())
        self.assertFalse(self.clean.duplicated().any())

    def test_text_columns_are_normalized_but_keys_are_untouched(self):
        self.assertEqual(sorted(self.clean['canal'].unique()), ['boutique', 'telephone', 'web'])
        self.assertEqual(sorted(self.clean['ville_livraison'].unique()), ['lyon', 'orleans', 'paris'])
        self.assertTrue(self.clean['id_client'].str.startswith(('C', '9')).all())
        self.assertIn('C02300', self.clean['id_client'].tolist(), 'les identifiants ne doivent pas être mis en minuscules')

    def test_missing_discount_becomes_zero(self):
        self.assertEqual(self.log['remises_manquantes_remplacees'], 1)
        self.assertFalse(self.clean['remise_pct'].isna().any())

    def test_amount_is_net_of_discount(self):
        row = self.clean[self.clean['id_commande'] == 'CMD1'].iloc[0]
        self.assertAlmostEqual(row['montant'], 3 * 160.66 * 0.95)

    def test_log_is_consistent_with_row_counts(self):
        removed = (self.log['doublons_supprimes'] + self.log['dates_manquantes_supprimees']
                   + self.log['dates_illisibles_supprimees'] + self.log['quantites_non_numeriques_supprimees']
                   + self.log['quantites_negatives_supprimees'])
        self.assertEqual(self.log['lignes_initiales'] - removed, self.log['lignes_finales'])

    def test_unusual_client_id_is_reported_not_removed(self):
        self.assertEqual(self.log['id_client_format_inhabituel'], 1)
        self.assertIn('990031', self.clean['id_client'].tolist())

    def test_idempotent(self):
        second, _ = clean_sales(self.clean, verbose=False)
        pd.testing.assert_frame_equal(second, self.clean)

    def test_max_quantity_option(self):
        capped, log = clean_sales(self.raw, max_quantity=2, verbose=False)
        self.assertTrue((capped['quantite'] <= 2).all())
        self.assertGreater(log['quantites_au_dela_du_max_supprimees'], 0)

    def test_keep_negative_quantities_when_asked(self):
        kept, log = clean_sales(self.raw, drop_negative_quantities=False, verbose=False)
        self.assertNotIn('quantites_negatives_supprimees', log)
        self.assertTrue((kept['quantite'] < 0).any())

    def test_missing_required_column_raises(self):
        with self.assertRaises(KeyError):
            clean_sales(self.raw.drop(columns=['prix_unitaire']), verbose=False)


if __name__ == '__main__':
    unittest.main()
