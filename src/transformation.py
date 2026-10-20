"""Transformation et encodage du jeu de ventes (séance 4).

Usage depuis un notebook situé à la racine du projet :

    import sys
    sys.path.insert(0, 'src')
    from transformation import scale_minmax, scale_zscore, prepare_features, fit_feature_params

Deux modes pour prepare_features :

  - jeu complet (export BI) : prepare_features(sales, products)
  - modélisation SANS fuite : on apprend les paramètres sur l'entraînement seulement,
    puis on les applique aux deux jeux.

        params = fit_feature_params(train, products)
        train_features = prepare_features(train, products, params=params)
        test_features = prepare_features(test, products, params=params)
"""
import numpy as np
import pandas as pd

__all__ = ['scale_minmax', 'scale_zscore', 'group_rare_categories',
           'fit_feature_params', 'prepare_features']

ONE_HOT_COLUMNS = {'canal': 'canal', 'statut': 'statut', 'categorie': 'cat'}  # colonne -> préfixe
FAILED_STATUSES = ['annule', 'retourne']   # statuts normalisés par clean_sales (séance 3)
RARE_FREQUENCY = 0.02

REQUIRED_SALES_COLUMNS = ['id_client', 'id_produit', 'quantite', 'prix_unitaire',
                          'remise_pct', 'montant', 'statut', 'canal']
REQUIRED_PRODUCT_COLUMNS = ['id_produit', 'categorie', 'sous_categorie']


# ---------------------------------------------------------------- mise à l'échelle

def scale_minmax(series, minimum=None, maximum=None):
    """(x - min) / (max - min). Sans argument, min et max sont ceux de la série.

    Pour mettre un jeu de test à l'échelle, passer le minimum et le maximum appris
    sur l'entraînement. Une série constante donne 0 (et non une division par zéro).
    """
    minimum = series.min() if minimum is None else minimum
    maximum = series.max() if maximum is None else maximum
    spread = maximum - minimum
    if spread == 0:
        return (series - minimum) * 0.0
    return (series - minimum) / spread


def scale_zscore(series, mean=None, std=None):
    """(x - moyenne) / écart type (corrigé, ddof=1, comme pandas).

    Pour un jeu de test, passer la moyenne et l'écart type appris sur l'entraînement.
    Une série constante donne 0 (et non une division par zéro).
    """
    mean = series.mean() if mean is None else mean
    std = series.std() if std is None else std
    if not std or np.isnan(std):
        return (series - mean) * 0.0
    return (series - mean) / std


# ---------------------------------------------------------------- encodage

def group_rare_categories(series, min_frequency=0.01, other_label='autre', keep=None):
    """Regroupe les modalités rares sous `other_label`. Les valeurs manquantes restent manquantes.

    `keep` : modalités à conserver, apprises sur l'entraînement. Si absent, elles sont
    déterminées à partir de `series` (fréquence >= min_frequency).
    """
    if keep is None:
        frequencies = series.value_counts(normalize=True)
        keep = frequencies[frequencies >= min_frequency].index
    return series.where(series.isin(keep) | series.isna(), other_label)


def _one_hot(series, categories, prefix):
    """One-hot sur des modalités FIXES : même colonnes en entraînement et en test.

    Une modalité inconnue donne une ligne de zéros au lieu de créer une colonne.
    """
    known = series.where(series.isin(categories))   # modalité inconnue -> manquante -> ligne de zéros
    categorical = pd.Series(pd.Categorical(known, categories=categories), index=series.index)
    return pd.get_dummies(categorical, prefix=prefix, prefix_sep='=', dtype=int)


# ---------------------------------------------------------------- préparation

def _check_columns(frame, required, label):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f'colonnes manquantes dans {label} : {missing}')


def _enrich(df, products):
    """Jointure produits + variables dérivées qui n'apprennent rien (pas de fuite possible)."""
    _check_columns(df, REQUIRED_SALES_COLUMNS, 'les ventes')
    _check_columns(products, REQUIRED_PRODUCT_COLUMNS, 'le référentiel produits')

    out = df.merge(products, on='id_produit', how='left',
                   validate='many_to_one', suffixes=('', '_prod'))

    # Marge (si le coût d'achat est connu). Un prix nul donnerait un taux infini : on le neutralise.
    if 'cout_achat' in out.columns:
        out['marge_unitaire'] = out['prix_unitaire'] - out['cout_achat']
        safe_price = out['prix_unitaire'].where(out['prix_unitaire'] > 0)
        out['taux_marge'] = (out['marge_unitaire'] / safe_price).round(4)

    # 1. Montant net de remise (remise_pct est en POURCENTS : 5 signifie 5 %)
    out['montant_net'] = (out['quantite'] * out['prix_unitaire']
                          * (1 - out['remise_pct'].fillna(0) / 100))

    # 3. Commande non aboutie : annulée ou retournée
    out['commande_non_aboutie'] = out['statut'].isin(FAILED_STATUSES).astype(int)
    return out


def fit_feature_params(df, products, min_frequency=RARE_FREQUENCY):
    """Apprend, sur `df` seulement, tout ce que prepare_features doit réutiliser sur un autre jeu."""
    return _learn_params(_enrich(df, products), min_frequency)


def _learn_params(enriched, min_frequency):
    frequencies = enriched['sous_categorie'].value_counts(normalize=True)
    souscat_keep = list(frequencies[frequencies >= min_frequency].index)
    souscat_grouped = group_rare_categories(enriched['sous_categorie'], keep=souscat_keep)

    categories = {column: sorted(enriched[column].dropna().unique()) for column in ONE_HOT_COLUMNS}
    categories['sous_categorie'] = sorted(souscat_grouped.dropna().unique())

    return {
        'montant_mean': enriched['montant'].mean(),
        'montant_std': enriched['montant'].std(),
        'client_mean_amount': enriched.groupby('id_client')['montant'].mean(),
        'price_median_by_category': enriched.groupby('categorie')['prix_unitaire'].median(),
        'price_median_global': enriched['prix_unitaire'].median(),
        'souscat_keep': souscat_keep,
        'categories': categories,
    }


def prepare_features(df, products, params=None, min_frequency=RARE_FREQUENCY):
    """Prépare le jeu de ventes : jointure produits, variables dérivées, encodage, z-score.

    Variables dérivées : montant_net, panier_moyen_client, commande_non_aboutie,
    ecart_prix_median_cat (plus marge_unitaire et taux_marge si cout_achat existe).

    params=None : les paramètres (moyennes, médianes, modalités retenues) sont appris sur `df`.
    params=fit_feature_params(train, products) : ils viennent de l'entraînement, ce qui
    évite la fuite de données quand `df` est le jeu de test.

    L'entrée n'est pas modifiée. Le nombre de lignes est conservé.
    """
    out = _enrich(df, products)
    if params is None:
        params = _learn_params(out, min_frequency)

    # 2. Panier moyen du client (un client inconnu de l'entraînement reste NaN)
    out['panier_moyen_client'] = out['id_client'].map(params['client_mean_amount'])

    # 4. Écart au prix médian de la catégorie (catégorie inconnue : médiane globale)
    category_median = out['categorie'].map(params['price_median_by_category'])
    category_median = category_median.fillna(params['price_median_global'])
    out['ecart_prix_median_cat'] = out['prix_unitaire'] - category_median

    # Encodage one-hot sur modalités fixes
    souscat_grouped = group_rare_categories(out['sous_categorie'], keep=params['souscat_keep'])
    dummies = [_one_hot(out[column], params['categories'][column], prefix)
               for column, prefix in ONE_HOT_COLUMNS.items()]
    dummies.append(_one_hot(souscat_grouped, params['categories']['sous_categorie'], 'souscat'))

    out = pd.concat([out.drop(columns=list(ONE_HOT_COLUMNS) + ['sous_categorie'])] + dummies, axis=1)

    # Mise à l'échelle z-score de montant, avec les paramètres appris
    out['montant_zscore'] = scale_zscore(out['montant'], params['montant_mean'], params['montant_std'])
    return out
