# Scripts

## Installation
Pour installer les dépendances: 
```bash
pip install -r requirements.txt
```

Mettre les scripts dans un dossier `script/`
Mettre les fichiers de données dans `data/`

Si besoin, utilisez `./besoin_client_{X}.py --help` pour obtenir de l'aide sur un package


## Besoin client 1 - Clustering Maritime

### Expérimental
Entraîne le modèle de clustering et génère `model.pkl`

Algorithmes supportés:
- KMeans
- Agglomerative 
- DBSCAN

Métriques d'évaluation:
- Méthode du coude (KMeans)
- Silhouette
- Calinski-Harabasz
- Davies-Bouldin

Score composite:
`0.4 * silhouette + 0.3 * calinski + 0.3 * davies`

Pour le choix de quel modèle prendre, il y a un input à l'utilisateur apres le résultat de chaque 

Fichiers générés:
- model.pkl
- analyse.png
- carte_clusters.html
- donnees_clusters.csv

### Utilisation
Utilise le modèle pour prédire le cluster d'un navire

Usage:
`./besoin_client_1.py --lat 26.13178 --lon -92.04043 --sog 12 --cog 12 --heading 12`
Ce navire est dans le cluster: 1
Est probablement dans un port dans la zone Houston/Nouvelle-Orléans

Paramètres:
--lat : Latitude
--lon : Longitude
--sog : Vitesse en noeud
--cog : Route
--heading : Cap 


## Besoin client 2 - ...
## Besoin client 2 - VesselType Prédiction

### Expérimental
Préparation d'un nouveau Dataframe de paramètres "avancés" -> Prédiction améliorée de VesselType

INPUT : - fichier nettoyé des données brutes AIS (.csv), fichier final des features améliorées (.csv)
OUTPUT : - PKL du meilleur modèle

Features additionnelles intégrées :
- mean-dist-travel
- std-dist-travel
- mean-duration-stop
- std-duration-stop
- mean-cruise-speed
- std-cruise-speed
- mean-dist-coast-travel
- std-dist-coast-travel
- mean-draft
- std-draft
- number-occurence-travel
- number-occurence-significative-draft-variation-onshore
- number-occurence-significative-draft-variation-offshore
- duration-onshore-ratio
- ratio-length-width
- ratio-directionnal-coherence

Preprocessing :
- Clippage des valeurs aberrantes avec IQR
- Remplacement des NA par la médiane (normalement il n'y en a pas)
- Vérification de la conformités des données
- Checking de la multicolinéarité (VIF)
- RobustScaler (normalisation adaptée à des outliers coriaces)

Comparaison entre les 3 algorithmes d'apprentissage supervisé suivant :
- LightGBM
- Random Forest
- XGBoost

Méthodes utilisées durant ou post entraînement :
- SMOTE (Équilibrage des classes)
- Cross-validation stratifié (Plusieurs plis pour avoir des stats sur la précision avec moins de facteur aléatoire)

### Utilisation
python script.py (pour exécuter le script)

3 manières de RUN (rajouter les paramètres suivants :
- --train --evaluate
- --predict --mmsi <mmsi_existant>
- --predict --features <value_var0, value_var1, ...>


## Besoin client 3 - Prédiction des trajets
### Expérimental

Après expérimentation des modèles:
- LinearRegression
- HistGradientBoosting
- RandomForest
- LSTM
- GRU

Le modèle le plus performant était le `RandomForest`
Cependant il demande beaucoup de stockage et de mémoire (24Gb)
Le modèle utilisé pour les prédictions par défaut est donc `LinearRegression`, plus léger et rapide

Pour des raisons de scaling, les scalers sont aussi importés en `.pkl`

### Utilisation
Utilise le modèle pour prédire les positions futures d'un navire

Usage:
`./besoin_client_3.py --steps 2 --json='{"LAT": 26.13178, "LON": -92.04043, "SOG": 12.0, "COG": 12.0, "Heading": 12.0, "VesselType": 31}'`
{"LAT": [26.13294624], "LON": [-92.03926376]}
{"LAT": [26.13480737], "LON": [-92.035186]}

Paramètres:
--json: données AIS du navire
--steps: nombre de positions à prédire (intervales de 5min)
