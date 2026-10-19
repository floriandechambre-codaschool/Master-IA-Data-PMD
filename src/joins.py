"""Outils de jointure sécurisée pour pandas."""


def safe_merge(left, right, **kwargs):
    """Jointure pandas qui échoue bruyamment si un left join change le nombre de lignes.

    Accepte les mêmes arguments que DataFrame.merge (on, how, validate, suffixes...).
    Affiche le taux de lignes de `left` sans correspondance dans `right`.
    """
    if 'indicator' in kwargs:
        raise ValueError("safe_merge gère déjà 'indicator' : ne le passez pas.")

    how = kwargs.get('how', 'inner')  # 'inner' est la valeur par défaut de pandas
    merged = left.merge(right, indicator='_merge_check', **kwargs)

    n_unmatched = (merged['_merge_check'] == 'left_only').sum()
    unmatched_rate = n_unmatched / len(merged) if len(merged) else 0.0
    merged = merged.drop(columns='_merge_check')

    print(f'safe_merge ({how}) : {len(left)} -> {len(merged)} lignes | '
          f'{n_unmatched} clés non appariées ({unmatched_rate:.2%})')

    if how == 'left' and len(merged) != len(left):
        raise ValueError(
            f"Le nombre de lignes a changé lors d'un left join : "
            f"{len(left)} -> {len(merged)}. La clé de droite n'est probablement pas unique."
        )
    return merged
