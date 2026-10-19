"""Nettoyage du jeu de ventes (séance 3).

Usage depuis un notebook situé à la racine du projet :

    import sys
    sys.path.insert(0, 'src')
    from cleaning import clean_sales

    clean_df, log = clean_sales(pd.read_csv('data/raw/ventes_brutes.csv'))

Principe : chaque décision de nettoyage est explicite et tracée dans le journal.
"""
import re

import numpy as np
import pandas as pd

__all__ = ['clean_sales', 'normalize_text', 'parse_price', 'parse_mixed_dates']

REQUIRED_COLUMNS = ['id_commande', 'date_commande', 'quantite', 'prix_unitaire',
                    'remise_pct', 'canal', 'statut', 'ville_livraison']

# Colonnes descriptives à normaliser. Les colonnes clés (id_*) ne le sont JAMAIS :
# les passer en minuscules casserait les jointures.
TEXT_COLUMNS = ['canal', 'statut', 'ville_livraison']

# Formats de date rencontrés dans ventes_brutes.csv, essayés dans cet ordre.
# Les formats avec jour/mois sont lus en jour d'abord (source française : « 24-05-2023 »
# prouve que le jour vient en premier pour les dates à tirets).
DATE_FORMATS = ['%Y/%m/%d %H:%M', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%d %b %Y']

CLIENT_ID_PATTERN = re.compile(r'^C\d{5}$')


def normalize_text(series):
    """Casse, espaces et accents : trois sources de faux niveaux. Les valeurs manquantes restent manquantes."""
    text = series.astype('string').str.strip().str.lower()
    text = text.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('utf-8')
    return text.str.replace(r'\s+', ' ', regex=True)


def parse_price(series):
    """Convertit une colonne de prix mixte (nombre ou texte '123,45 EUR') en float.

    Les valeurs illisibles deviennent NaN (elles sont comptées par clean_sales).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype('string').str.replace(r'[^\d,.\-]', '', regex=True)
    return pd.to_numeric(text.str.replace(',', '.', regex=False), errors='coerce').astype(float)


def parse_mixed_dates(series, formats=DATE_FORMATS):
    """Convertit une colonne de dates saisies dans plusieurs formats.

    pd.to_datetime sans format fige le format de la première valeur et rejette tout le
    reste : sur une colonne mixte, on perd l'essentiel des lignes sans erreur. On essaie
    donc chaque format connu sur les valeurs encore non lues, puis on tente une lecture
    souple (jour d'abord) pour ce qui reste. Une colonne déjà en datetime est rendue telle quelle.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    text = series.astype('string').str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')

    for fmt in formats:
        todo = (result.isna() & text.notna()).to_numpy()
        if not todo.any():
            break
        parsed = pd.to_datetime(text[todo], format=fmt, errors='coerce')
        result.iloc[np.flatnonzero(todo)] = parsed.to_numpy()

    todo = (result.isna() & text.notna()).to_numpy()
    if todo.any():
        parsed = pd.to_datetime(text[todo], format='mixed', dayfirst=True, errors='coerce')
        result.iloc[np.flatnonzero(todo)] = parsed.to_numpy()
    return result


def clean_sales(df, drop_negative_quantities=True, max_quantity=None, verbose=True):
    """Nettoie le jeu de ventes et renvoie (df_propre, log).

    Étapes, dans l'ordre :
      1. prix_unitaire en nombre (virgule décimale et suffixe EUR gérés) ;
      2. normalisation des colonnes descriptives (canal, statut, ville_livraison) ;
      3. suppression des doublons stricts, après normalisation ;
      4. dates converties (formats mixtes), lignes sans date exploitable supprimées ;
      5. remise_pct manquante remplacée par 0 ;
      6. quantités : non numériques supprimées, négatives supprimées (optionnel),
         au-delà de max_quantity supprimées (optionnel) ;
      7. montant = quantite x prix_unitaire x (1 - remise_pct / 100).

    La fonction est idempotente : l'appliquer à son propre résultat ne change rien.
    L'entrée n'est jamais modifiée. Le journal compte chaque modification.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f'colonnes manquantes : {missing}')

    log = {'lignes_initiales': int(len(df))}
    df = df.copy()

    # 1. prix_unitaire -> numérique
    df['prix_unitaire'] = parse_price(df['prix_unitaire'])
    log['prix_unitaire_illisibles'] = int(df['prix_unitaire'].isna().sum())

    # 2. normaliser les colonnes descriptives (jamais les clés)
    for column in TEXT_COLUMNS:
        before = df[column].nunique()
        df[column] = normalize_text(df[column])
        log[f'normalisation_{column}'] = f'{before} → {df[column].nunique()} modalités'

    # 3. doublons stricts (après normalisation : elle en révèle de nouveaux)
    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log['doublons_supprimes'] = n_duplicates

    # 4. dates -> datetime, suppression des lignes sans date exploitable
    was_missing = df['date_commande'].isna()
    df['date_commande'] = parse_mixed_dates(df['date_commande'])
    unreadable = df['date_commande'].isna() & ~was_missing
    log['dates_manquantes_supprimees'] = int(was_missing.sum())
    log['dates_illisibles_supprimees'] = int(unreadable.sum())
    df = df.dropna(subset=['date_commande'])

    # 5. remise_pct : une remise non saisie vaut 0 (stratégie 'fill_zero')
    df['remise_pct'] = pd.to_numeric(df['remise_pct'], errors='coerce')
    log['remises_manquantes_remplacees'] = int(df['remise_pct'].isna().sum())
    df['remise_pct'] = df['remise_pct'].fillna(0)

    # 6. quantités
    df['quantite'] = pd.to_numeric(df['quantite'], errors='coerce')
    log['quantites_non_numeriques_supprimees'] = int(df['quantite'].isna().sum())
    df = df.dropna(subset=['quantite'])

    if drop_negative_quantities:
        negative = df['quantite'] < 0
        log['quantites_negatives_supprimees'] = int(negative.sum())
        df = df[~negative]

    if max_quantity is not None:
        too_large = df['quantite'] > max_quantity
        log['quantites_au_dela_du_max_supprimees'] = int(too_large.sum())
        df = df[~too_large]

    # 7. montant net de remise (NaN si le prix est illisible)
    df['montant'] = df['quantite'] * df['prix_unitaire'] * (1 - df['remise_pct'] / 100)
    log['montants_non_calculables'] = int(df['montant'].isna().sum())

    # Contrôle informatif, sans modification : identifiants clients hors format
    if 'id_client' in df.columns:
        bad_ids = ~df['id_client'].astype('string').str.match(CLIENT_ID_PATTERN).fillna(False)
        log['id_client_format_inhabituel'] = int(bad_ids.sum())

    df = df.reset_index(drop=True)
    log['lignes_finales'] = int(len(df))

    if verbose:
        for key, value in log.items():
            print(f'  {key:<40} {value}')

    return df, log
