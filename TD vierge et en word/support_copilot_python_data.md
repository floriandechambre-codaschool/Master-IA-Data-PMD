# Support de cours Python / Data — format lisible par Copilot

Document consolidé à partir des notebooks fournis. Le contenu est organisé par séance, atelier, explication, code et exercices.


# Séance 1 — Environnement de travail et NumPy

**Source :** `01_environnement_numpy.ipynb`

# Séance 1 — Environnement de travail et NumPy

## Ce que vous saurez faire à la fin

- créer un environnement virtuel reproductible et figer ses dépendances ;
- manipuler des tableaux NumPy : slicing, masques booléens, statistiques par axe ;
- mesurer et expliquer l'écart de performance entre une boucle Python et une opération vectorisée.

> **Convention du module.** Le dossier `data/raw/` est en lecture seule. Toute transformation
> produit un nouveau fichier dans `data/processed/`. Aucune donnée brute n'est jamais écrasée.

> **Convention de nommage du module.** Le code est écrit en anglais et suit la PEP 8 :
> fonctions et variables en `snake_case`, constantes en `MAJUSCULES`. Les **noms de colonnes**
> restent en français parce qu'ils viennent de la source : renommer les colonnes d'un fichier
> d'entrée est une transformation comme une autre, elle se décide et se documente, elle ne se
> fait pas par réflexe. Vous rencontrerez cette situation partout en entreprise.

## Préparation de la session


```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)

print('Données disponibles :')
for path in sorted(RAW_DIR.glob('*')):
    print(' ', path.name)


# Parquet conserve les types (dates, entiers, catégories) là où le CSV les perd :
# c'est le format à privilégier entre deux étapes d'un pipeline. Repli automatique
# sur le CSV si pyarrow n'est pas installé.
def save_dataset(df, name):
    try:
        path = PROCESSED_DIR / f'{name}.parquet'
        df.to_parquet(path, index=False)
    except ImportError:
        path = PROCESSED_DIR / f'{name}.csv'
        df.to_csv(path, index=False)
        print('(pyarrow absent : repli sur le CSV)')
    print('écrit :', path.name, df.shape)
    return path


def load_dataset(name):
    parquet_path = PROCESSED_DIR / f'{name}.parquet'
    csv_path = PROCESSED_DIR / f'{name}.csv'
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f'{name} introuvable : exécutez le notebook précédent')


def dataset_exists(name):
    return ((PROCESSED_DIR / f'{name}.parquet').exists()
            or (PROCESSED_DIR / f'{name}.csv').exists())
```

## Atelier 1.1 — Environnement et projet (30 min)

# 1. Créer et activer l'environnement

# 2. Installer les dépendances

# 3. Figer l'état exact de l'environnement

# 4. Versionner le projet

git init
printf '.venv/\ndata/raw/\n__pycache__/\n.ipynb_checkpoints/\n' > .gitignore
git add . && git commit -m 'Initialisation du projet PMD'
```

**Pourquoi `data/raw/` dans le `.gitignore` ?** Les données brutes peuvent être
volumineuses et ne changent jamais. On versionne le *code qui les transforme*, pas
les données elles-mêmes. Le `README` doit indiquer où les récupérer.


```python
# Vérification : l'interpréteur utilisé est-il bien celui de l'environnement virtuel ?
import sys

print('Interpréteur :', sys.executable)
print('Version      :', sys.version.split()[0])

for package_name in ['numpy', 'pandas', 'sklearn', 'matplotlib']:
    try:
        module = __import__(package_name)
        print(f'{package_name:<12} {getattr(module, "__version__", "?")}')
    except ImportError:
        print(f'{package_name:<12} ABSENT')
```


```python
!git add .
!git commit -m "Initialisation du projet PMD"
```

**À vérifier avant de continuer.** Le chemin de l'interpréteur doit contenir `.venv`.
S'il pointe vers le Python système, le noyau du notebook n'est pas celui de votre
environnement : installez `ipykernel` dans le venv et sélectionnez le bon noyau.

## Atelier 1.2 — NumPy : le tableau comme unité de calcul (50 min)

### Partie guidée

Un `ndarray` a un **type unique** pour toutes ses valeurs. C'est ce qui permet à NumPy de
stocker les données de façon contiguë en mémoire et d'appliquer une opération à l'ensemble
du tableau sans boucle Python.


```python
import numpy as np
rng = np.random.default_rng(42)

# 12 relevés de température pour 5 capteurs : matrice 5 lignes x 12 colonnes
temperatures = np.round(rng.normal(loc=14, scale=6, size=(5, 12)), 1)

print('forme   :', temperatures.shape)
print('type    :', temperatures.dtype)
print('mémoire :', temperatures.nbytes, 'octets')
temperatures
```


```python
# Statistiques par axe : axis=0 parcourt les lignes, axis=1 parcourt les colonnes
print('Moyenne par mois (sur les 5 capteurs) :', np.round(temperatures.mean(axis=0), 2))
print('Moyenne par capteur (sur les 12 mois) :', np.round(temperatures.mean(axis=1), 2))
print('Moyenne globale                       :', round(temperatures.mean(), 2))
```

**L'axe est la dimension qui disparaît.** `axis=0` agrège sur les lignes et laisse
12 valeurs, une par colonne. C'est la source d'erreur numéro un chez les débutants.


```python
# Masque booléen : un tableau de True/False de même forme que l'original
is_freezing = temperatures < 0

print('Nombre de relevés sous zéro :', is_freezing.sum())
print('Capteurs concernés          :', np.where(is_freezing.any(axis=1))[0])

# Le masque sert à lire...
print('Valeurs négatives           :', temperatures[is_freezing])

# ... et à écrire
corrected = temperatures.copy()
corrected[is_freezing] = 0
print('Minimum après correction    :', corrected.min())
```

### Partie autonome

Complétez les cellules suivantes. Chaque cellule se termine par un `assert` :
s'il passe sans message, votre réponse est correcte.


```python
# Q1. Construire un tableau des écarts de chaque relevé à la moyenne de SON capteur.
#     Attendu : forme (5, 12), et la moyenne de chaque ligne doit valoir 0.
#     Indice : temperatures.mean(axis=1) a la forme (5,) ; il faut la remettre en (5, 1)
#     pour que le broadcasting aligne les lignes. Voir .reshape(-1, 1) ou [:, None].

# Moyenne par capteur, remise en colonne
means = temperatures.mean(axis=1).reshape(-1, 1)

# Écarts à la moyenne
deviations = temperatures - means

# Vérifications
assert deviations.shape == (5, 12), 'forme incorrecte'
assert np.allclose(deviations.mean(axis=1), 0), 'les lignes ne sont pas centrées'

print('OK — écart type des écarts :', round(deviations.std(), 2))
```


```python
# Q2. Compter, pour chaque capteur, le nombre de mois où la température dépasse
#     la moyenne globale de tout le tableau.
#     Attendu : un tableau de 5 entiers.

months_above_mean = ... # TODO

global_mean = temperatures.mean()
mask = temperatures > global_mean
months_above_mean = mask.sum(axis=1)

assert months_above_mean.shape == (5,), 'un compte par capteur est attendu'
assert months_above_mean.sum() == (temperatures > temperatures.mean()).sum()
print('OK —', months_above_mean)
```


```python
# Q3. Remplacer les valeurs aberrantes par la médiane du tableau.
#     Est aberrante toute valeur à plus de 2 écarts types de la moyenne globale.
#     Travaillez sur une COPIE : le tableau d'origine ne doit pas changer.

cleaned = ...  # TODO

assert cleaned is not temperatures, 'vous avez modifié le tableau original'
assert cleaned.shape == temperatures.shape
threshold = 2 * temperatures.std()
assert np.abs(cleaned - temperatures.mean()).max() <= threshold + 1e-9
print('OK — valeurs remplacées :', int((cleaned != temperatures).sum()))
```

## Atelier 1.3 — Pourquoi vectoriser (40 min)

L'argument est souvent présenté comme une question de style. C'est une question d'ordre
de grandeur. Mesurez-le vous-même.


```python
import time


def benchmark(func, *args, n_repeats=3):
    """Renvoie (meilleur temps en secondes, résultat de la fonction)."""
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        result = func(*args)
        durations.append(time.perf_counter() - start)
    return min(durations), result


values = rng.normal(size=2_000_000)
print('Tableau de', f'{values.size:,}'.replace(',', ' '), 'valeurs')
```


```python
def sum_squares_loop(x):
    total = 0.0
    for value in x:
        total += value * value
    return total


def sum_squares_comprehension(x):
    return sum(value * value for value in x)


def sum_squares_numpy(x):
    return np.sum(x ** 2)


timings = {}
for label, func in [('boucle for', sum_squares_loop),
                    ('compréhension', sum_squares_comprehension),
                    ('numpy vectorisé', sum_squares_numpy)]:
    duration, result = benchmark(func, values)
    timings[label] = duration
    print(f'{label:<18} {duration:7.4f} s   (résultat {result:.2f})')

baseline = timings['numpy vectorisé']
print()
for label, duration in timings.items():
    print(f'{label:<18} x{duration / baseline:6.1f} par rapport à NumPy')
```


```python
# Q4. Même comparaison pour un filtrage : compter les valeurs supérieures à 1,5.
#     Écrivez les deux versions et comparez.

def count_above_loop(x, threshold=1.5):
    ...  # TODO


def count_above_numpy(x, threshold=1.5):
    ...  # TODO


loop_duration, loop_result = benchmark(count_above_loop, values)
numpy_duration, numpy_result = benchmark(count_above_numpy, values)

assert loop_result == numpy_result, 'les deux versions ne donnent pas le même résultat'
print(f'boucle {loop_duration:.4f} s | numpy {numpy_duration:.4f} s '
      f'| facteur x{loop_duration / numpy_duration:.0f}')
```

### À retenir

L'écart typique est d'un facteur 30 à 100. Il ne vient pas de la vitesse du calcul lui-même
mais du coût de l'interprétation : chaque tour de boucle Python crée des objets, vérifie des
types et appelle des méthodes. NumPy délègue la boucle à du code compilé qui travaille
directement sur un bloc mémoire homogène.

**Conséquence pratique pour le reste du module :** dès que vous écrivez `for` sur les lignes
d'un DataFrame, arrêtez-vous et cherchez l'opération vectorisée équivalente.

## Livrable de la séance

Un dépôt Git contenant :

- `requirements.txt` généré par `pip freeze` ;
- `.gitignore` excluant `.venv/` et `data/raw/` ;
- ce notebook exécuté, avec les quatre questions complétées ;
- un `README.md` de dix lignes décrivant le projet et la procédure d'installation.


---


# Séance 2 — Pandas : structures et exploration

**Source :** `02_pandas_exploration.ipynb`

# Séance 2 — Pandas : structures et exploration

## Ce que vous saurez faire à la fin

- charger une même source depuis trois formats et réconcilier les types obtenus ;
- sélectionner et filtrer sans déclencher de `SettingWithCopyWarning` ;
- produire automatiquement un rapport d'exploration sur un jeu de données inconnu.

Le jeu de travail est `ventes_brutes.csv` : environ 17 600 lignes de commandes,
volontairement dégradées. Vous allez le retrouver à chaque séance jusqu'à la fin du module.

> **Convention de nommage du module.** Le code est écrit en anglais et suit la PEP 8 :
> fonctions et variables en `snake_case`, constantes en `MAJUSCULES`. Les **noms de colonnes**
> restent en français parce qu'ils viennent de la source : renommer les colonnes d'un fichier
> d'entrée est une transformation comme une autre, elle se décide et se documente, elle ne se
> fait pas par réflexe. Vous rencontrerez cette situation partout en entreprise.


```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)

print('Données disponibles :')
for path in sorted(RAW_DIR.glob('*')):
    print(' ', path.name)


# Parquet conserve les types (dates, entiers, catégories) là où le CSV les perd :
# c'est le format à privilégier entre deux étapes d'un pipeline. Repli automatique
# sur le CSV si pyarrow n'est pas installé.
def save_dataset(df, name):
    try:
        path = PROCESSED_DIR / f'{name}.parquet'
        df.to_parquet(path, index=False)
    except ImportError:
        path = PROCESSED_DIR / f'{name}.csv'
        df.to_csv(path, index=False)
        print('(pyarrow absent : repli sur le CSV)')
    print('écrit :', path.name, df.shape)
    return path


def load_dataset(name):
    parquet_path = PROCESSED_DIR / f'{name}.parquet'
    csv_path = PROCESSED_DIR / f'{name}.csv'
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f'{name} introuvable : exécutez le notebook précédent')


def dataset_exists(name):
    return ((PROCESSED_DIR / f'{name}.parquet').exists()
            or (PROCESSED_DIR / f'{name}.csv').exists())
```

## Atelier 2.1 — Lire une source, vraiment (50 min)

### Partie guidée : la lecture naïve et ce qu'elle cache


```python
sales = pd.read_csv(RAW_DIR / 'ventes_brutes.csv')

print('Dimensions :', sales.shape)
sales.head()
```


```python
sales.dtypes
```

Regardez `prix_unitaire`. Le type est `object`, autrement dit du texte, alors qu'il s'agit
d'un montant. Cherchons pourquoi.


```python
# Isoler les valeurs qui ne se convertissent pas en nombre
as_number = pd.to_numeric(sales['prix_unitaire'], errors='coerce')
unparsable = sales.loc[as_number.isna(), 'prix_unitaire']

print('Valeurs non convertibles :', len(unparsable))
print(unparsable.head(8).tolist())
```

Une partie des prix a été saisie avec une virgule décimale et un symbole monétaire.
Une lecture qui ignore ce détail produit une colonne texte, et tout calcul en aval échoue
silencieusement ou renvoie une erreur bien plus loin dans le pipeline.

**C'est la règle à retenir : ne jamais faire confiance à l'inférence de types.** Vérifiez
systématiquement `dtypes` après chaque lecture.


```python
def parse_price(series):
    """Convertit une colonne de prix mixte (nombre ou texte '123,45 EUR') en float."""
    text = series.astype(str).str.replace(' EUR', '', regex=False)
    text = text.str.replace(',', '.', regex=False).str.strip()
    return pd.to_numeric(text, errors='coerce')


unit_price = parse_price(sales['prix_unitaire'])
print('Valeurs encore non convertibles :', unit_price.isna().sum())
print(unit_price.describe().round(2))
```

### Partie autonome


```python
# Q1. Charger l'extrait des ventes depuis les trois formats disponibles,
#     puis comparer les dtypes obtenus pour la colonne 'date_commande'.



sample_csv = pd.read_csv(RAW_DIR /'ventes_brutes.csv')
sample_xlsx = pd.read_excel(RAW_DIR /'ventes_extrait.xlsx')
sample_json = pd.read_json(RAW_DIR /'ventes_extrait.json')

for label, frame in [('csv', sample_csv), ('xlsx', sample_xlsx), ('json', sample_json)]:
    print(f"{label:<6} lignes={len(frame):<6} dtype date={frame['date_commande'].dtype}")
```

**Question à traiter par écrit dans la cellule suivante.** Les trois formats donnent-ils
le même type pour `date_commande` ? Le même nombre de lignes ? Que se passerait-il si votre
pipeline acceptait indifféremment ces trois sources ?

*Votre réponse :*


```python
# Q2. Relire ventes_brutes.csv en une seule instruction, en imposant :
#     - id_client, id_produit, id_magasin comme chaînes de caractères ;
#     - la chaîne vide et 'NC' comme valeurs manquantes ;
#     - seulement les colonnes id_commande, date_commande, id_produit, quantite,
#       prix_unitaire, statut.
#     Indice : paramètres dtype, na_values et usecols.

EXPECTED_COLUMNS = ['id_commande', 'date_commande', 'id_produit',
                    'quantite', 'prix_unitaire', 'statut']

sales_subset = pd.read_csv(
    RAW_DIR / 'ventes_brutes.csv',
    dtype={'id_client': str, 'id_produit': str, 'id_magasin': str},
    na_values=['', 'NC'],
    usecols=EXPECTED_COLUMNS
)

assert list(sales_subset.columns) == EXPECTED_COLUMNS
assert sales_subset['id_produit'].dtype == object
print('OK —', sales_subset.shape)
```

## Atelier 2.2 — Sélection et filtrage (60 min)

### Partie guidée : `loc`, `iloc` et le piège de la copie


```python
sales['prix_unitaire'] = parse_price(sales['prix_unitaire'])
sales['montant'] = sales['quantite'] * sales['prix_unitaire']

# loc travaille sur les ÉTIQUETTES (noms de colonnes, valeurs d'index)
print(sales.loc[0:2, ['id_commande', 'quantite', 'montant']])
print()
# iloc travaille sur les POSITIONS entières
print(sales.iloc[0:2, [0, 5, -1]])
```

Notez la différence sur les bornes : `loc[0:2]` renvoie **trois** lignes (borne incluse),
`iloc[0:2]` en renvoie **deux** (borne exclue, comme le slicing Python).


```python
# Filtrage booléen : combiner des conditions avec & et |, chaque condition entre parenthèses
large_orders = sales[(sales['montant'] > 1000) & (sales['statut'] == 'livre')]
print('Commandes livrées de plus de 1000 EUR :', len(large_orders))

# query() est souvent plus lisible quand les conditions s'accumulent
same_result = sales.query("montant > 1000 and statut == 'livre'")
print('Même résultat :', len(same_result) == len(large_orders))
```


```python
# LE PIÈGE : modifier un sous-ensemble extrait par filtrage
cancelled = sales[sales['statut'] == 'annule']

# La ligne suivante déclenche un SettingWithCopyWarning : pandas ne sait pas si
# `cancelled` est une vue sur `sales` ou une copie indépendante.
cancelled['montant'] = 0

print('Montant dans sales pour les annulées :',
      sales.loc[sales['statut'] == 'annule', 'montant'].head(3).tolist())
print("-> la modification n'a PAS été propagée : le travail est perdu")
```

# Intention A : modifier le DataFrame d'origine

# Intention B : travailler sur une copie indépendante

cancelled = sales[sales['statut'] == 'annule'].copy()
cancelled['montant'] = 0
```

Le message d'avertissement est le symptôme d'une ambiguïté dans votre code, pas un bruit
à faire taire.

### Partie autonome


```python
# Q3. Extraire les commandes qui remplissent TOUTES ces conditions :
#     - statut 'livre' ;
#     - quantite comprise entre 1 et 10 inclus ;
#     - montant strictement supérieur à la médiane des montants des commandes livrées ;
#     - canal contenant 'web', quelle que soit la casse.
#     Le résultat doit être une COPIE indépendante.

target_orders = ...  # TODO

assert isinstance(target_orders, pd.DataFrame)
assert target_orders['quantite'].between(1, 10).all()
assert (target_orders['statut'] == 'livre').all()
assert target_orders['canal'].str.strip().str.lower().eq('web').all()
print('OK —', len(target_orders), 'commandes retenues')
```


```python
# Q4. Sans utiliser groupby, calculer le montant total des commandes livrées
#     pour chacun des trois canaux (après normalisation de la casse).
#     Un dictionnaire {canal: total} est attendu.

revenue_by_channel = {}  # TODO

assert set(revenue_by_channel) == {'web', 'boutique', 'telephone'}
print('OK')
for channel, total in sorted(revenue_by_channel.items(), key=lambda item: -item[1]):
    print(f'  {channel:<12} {total:>14,.2f} EUR'.replace(',', ' '))
```

## Atelier 2.3 — Un rapport d'exploration réutilisable (100 min)

### Partie guidée : les briques


```python
print('--- shape ---'); print(sales.shape)
print('\n--- info ---'); sales.info(memory_usage='deep')
```


```python
print('--- taux de valeurs manquantes ---')
missing_rate = (sales.isna().mean() * 100).round(2).sort_values(ascending=False)
print(missing_rate[missing_rate > 0])

print('\n--- cardinalité des colonnes texte ---')
for column in sales.select_dtypes(include='object').columns:
    print(f'  {column:<18} {sales[column].nunique():>6} modalités')
```


```python
print('--- modalités de canal, telles quelles ---')
print(sales['canal'].value_counts())
```

Neuf modalités pour ce qui devrait en compter trois. La casse et les espaces parasites
créent de faux niveaux. Un `value_counts()` brut est précisément l'outil qui révèle ce
genre de problème : c'est pourquoi il doit figurer dans le rapport automatique.

### Partie autonome : la fonction `profile_dataframe`


```python
def profile_dataframe(df, name='jeu de données', max_cardinality=25):
    """Affiche un rapport d'exploration standard.

    Doit produire, dans cet ordre :
      1. le nom, les dimensions et l'empreinte mémoire ;
      2. le nombre de lignes strictement dupliquées ;
      3. un tableau par colonne : type, nombre de valeurs manquantes, taux en %,
         nombre de valeurs distinctes ;
      4. les statistiques descriptives des colonnes numériques ;
      5. pour chaque colonne texte de cardinalité inférieure à max_cardinality,
         la répartition des modalités.

    Ne renvoie rien : la fonction affiche.
    """
    # TODO
    ...


profile_dataframe(sales, name='ventes_brutes')
```


```python
# Q5. Appliquer la fonction aux deux autres jeux de données et vérifier qu'elle
#     se comporte correctement sur des structures différentes.

customers = pd.read_csv(RAW_DIR / 'clients.csv')
sensors = pd.read_csv(RAW_DIR / 'capteurs.csv')

profile_dataframe(customers, name='clients')
profile_dataframe(sensors, name='capteurs')
```


```python
# Q6. Déplacer la fonction dans src/exploration.py, puis l'importer ici.
#     Le notebook doit rester lisible : le code réutilisable vit dans src/.

import sys
sys.path.insert(0, str(ROOT / 'src'))

# from exploration import profile_dataframe   # décommentez une fois le fichier créé
```

## Ce que le rapport révèle déjà

Rédigez ici, en cinq à dix lignes, la liste des anomalies que votre rapport a mises au jour
sur `ventes_brutes`. Ce texte est le point de départ de la séance 3 et le premier élément
de votre note méthodologique de projet.

*Vos observations :*

1. 
2. 
3.

## Livrable de la séance

- `src/exploration.py` contenant la fonction `profile_dataframe`, importable ;
- ce notebook exécuté, avec les six questions complétées ;
- la liste écrite des anomalies constatées.
