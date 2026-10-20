"""Contrôle d'intégrité référentielle d'un modèle en étoile."""


def check_referential_integrity(fact_table, dimensions):
    """dimensions : dict {nom_colonne_cle: dataframe_dimension}

    Affiche, pour chaque clé, le nombre de valeurs de la table de faits absentes
    de la dimension, puis lève une exception si au moins une clé est orpheline.
    """
    failures = {}

    for key, dim in dimensions.items():
        if key not in fact_table.columns:
            raise KeyError(f"Clé '{key}' absente de la table de faits")
        if key not in dim.columns:
            raise KeyError(f"Clé '{key}' absente de la dimension")
        if not dim[key].is_unique:
            raise ValueError(f"La clé '{key}' n'est pas unique dans la dimension")

        is_orphan = ~fact_table[key].isin(dim[key])
        n_orphans = int(is_orphan.sum())
        print(f'{key:<12} {n_orphans:>5} clé(s) orpheline(s) '
              f'({n_orphans / len(fact_table):.2%} des lignes)')

        if n_orphans:
            failures[key] = fact_table.loc[is_orphan, key].unique()[:5].tolist()

    if failures:
        details = '; '.join(f'{key} : {values}' for key, values in failures.items())
        raise AssertionError(f'Intégrité référentielle violée -> exemples : {details}')

    print('OK : toutes les clés de la table de faits existent dans leur dimension.')
