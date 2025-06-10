# Scratchpad (Robin)

Notes :
- J'ai réglé tout les problèmes de Warning à l'update
- J'ai réglé les warning à l'ouverture du terminal R & j'ai résolu le problème des libs qui s'installe pas (normalement) -> VIM installed
- Navire militaire moins d'infos peut-être ("NA") ? À checker -> PAS DE NAVIRE MILITAIRE DONC NON
- Uniquement un seul 5, deux 99 (navire immobiles) et un seul 31, le reste c'est entre 60 et 89
- Détection de routes commerciales et routes de croisières ? -> #NOÉ

********************************************************************************************************************

Advanced Features Engineering -> Derivated Features :
- Pour les Passengers -> vitesse de croisière récurrente ? vitesse moyenne stable ? rapport avec la taille car les bateaux ? routes récurrentes ? 
- Type de port -> si arrêt proche des arrêts des autres Tankers, Cargos ou Passengers #NOÉ
- Cohérence directionnelle entre COG et Heading ? -> On peut voir des tendances se dessigner qui différentie selon VesselType peut-être ? 
- Delta par rapport aux cotes

********************************************************************************************************************

Enhanced ML ? :
- 4 modèles possiblement à test : RandomForest, RegLog, XGBoost, SVM -> Combinaison des 4 (comparaison des résultats, pondération différente ?) 
    + Test avec plusieurs hyperparamètres ?
- Gestion des déséquilibres des classes -> Si dans le sample, on a backtest sur 1000 Cargo, 100 Tanker et 10 Passengers -> il va dire Cargo quand il sait pas trop -> voir comment équilibrer
    -> Voir solution SMOTE
- Validation croisée à plusieurs plis -> en gros plusieurs train/test pour éviter des backtest malchanceux ou chanceux

********************************************************************************************************************

-> Delta important entre la précision lors du train/test et lors du test "propre" sur des prédictions indépendantes : Écart de ~45% ?
    Explication : 
        - Overfitting ? -> train/test sur les mêmes données lors de evaluate_combinaison() -> SOLUTION : ?
        - Jeu de classes déséquilibrés + Apprentissage de la classe majoritaire ? -> SOLUTION : Stratifier les données
        - Meilleure combinaison -> SOG, Cargo, peut-être le hasard, c'est quand même très faible, et pas très discriminant -> SOLUTION : Soit variables à forte logique, soit nombre minimal de variables

********************************************************************************************************************

DONE : 
- On shore et Off shore pour Draft Variation
- Transceiver B corrélation
- Segmentation pour considérer les données comme des trajectoires de navires et donc pouvoir utiliser LON, LAT et SOG réellement (vitesse de croisière, durée d'arrêt, distance parcouru)

À partir de là -> Tester la précision du modèle jusqu'à aboutir à un résultat autrement plus satisfaisant
