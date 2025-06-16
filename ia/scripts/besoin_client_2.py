#!/usr/bin/env python3

"""
Pipeline Machine Learning pour la Prédiction du Type de Navire
================================================================

Ce script implémente un pipeline complet de machine learning pour prédire
le type de navire (VesselType) basé sur les caractéristiques des MMSI.

Utilisation:
    python vessel_prediction.py --train --evaluate
    python vessel_prediction.py --predict --mmsi 209016000
    python vessel_prediction.py --predict --features 8.3,2.4,345.3,570.5,8.5,5.4,...
"""

import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VesselTypePredictor:
    """
    Classe principale pour la prédiction du type de navire.
    
    Cette classe encapsule tout le pipeline de machine learning :
    - Préparation et prétraitement des données
    - Entraînement de multiples modèles
    - Évaluation et sélection du meilleur modèle
    - Prédiction sur de nouvelles données
    """
    
    def __init__(self):
        self.feature_columns = [
            'mean_draft', 'std_draft', 'mean_distance_travel', 'std_distance_travel',
            'mean_cruise_speed', 'std_cruise_speed', 'mean_dist_coast_travel',
            'std_dist_coast_travel', 'mean_duration_stop', 'std_duration_stop',
            'number_occurence_travel', 'length_width_ratio', 'COG_std',
            'HDG_COG_diff_mean', 'HDG_COG_diff_std', 'Heading_consistency',
            'COG_consistency', 'number_occurence_docked', 'number_occurence_off_shore_docked',
            'onshore_duration_ratio'
        ]
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.label_encoder = None
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def vessel_type_to_category(self, vessel_type: float) -> str:
        """
        Convertit les codes VesselType en catégories.
        
        Args:
            vessel_type: Code numérique du type de navire
            
        Returns:
            str: Catégorie ('Passenger', 'Cargo', 'Tanker', 'Other')
        """
        if pd.isna(vessel_type):
            return 'Other'
        
        vessel_type = int(vessel_type)
        if 60 <= vessel_type <= 69:
            return 'Passenger'
        elif 70 <= vessel_type <= 79:
            return 'Cargo'
        elif 80 <= vessel_type <= 89:
            return 'Tanker'
        else:
            return 'Other'
    
    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données en supprimant les valeurs infinies et aberrantes.
        
        Args:
            X: DataFrame des features
            
        Returns:
            pd.DataFrame: DataFrame nettoyé
        """
        logger.info("Nettoyage des données...")
        
        X_clean = X.copy()
        
        # Remplacer les valeurs infinies par NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Statistiques avant nettoyage
        inf_count = np.isinf(X.values).sum()
        nan_count = X.isnull().sum().sum()
        logger.info(f"Avant nettoyage - Valeurs infinies: {inf_count}, NaN: {nan_count}")
        
        # Pour chaque colonne, remplacer les NaN par la médiane
        for col in X_clean.columns:
            if X_clean[col].isnull().sum() > 0:
                median_val = X_clean[col].median()
                X_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Colonne {col}: {X_clean[col].isnull().sum()} NaN remplacés par médiane ({median_val:.2f})")
        
        # Détecter et traiter les valeurs aberrantes avec IQR
        for col in X_clean.columns:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Définir les limites (plus conservateur que 1.5*IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Compter les outliers
            outliers = ((X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Clip les valeurs extrêmes
                X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
                logger.info(f"Colonne {col}: {outliers} valeurs aberrantes clippées")
        
        # Vérification finale
        inf_count_after = np.isinf(X_clean.values).sum()
        nan_count_after = X_clean.isnull().sum().sum()
        logger.info(f"Après nettoyage - Valeurs infinies: {inf_count_after}, NaN: {nan_count_after}")
        
        return X_clean
    
    def load_and_prepare_data(self, features_file: str, raw_data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge et prépare les données pour l'entraînement.
        
        Args:
            features_file: Chemin vers le fichier des features calculées
            raw_data_file: Chemin vers le fichier des données brutes
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et labels
        """
        logger.info("Chargement des données...")
        
        # Charger les features
        features_df = pd.read_csv(features_file)
        
        # Charger les données brutes pour récupérer VesselType
        raw_df = pd.read_csv(raw_data_file)
        
        # Merger sur MMSI pour récupérer VesselType
        merged_df = features_df.merge(
            raw_df[['MMSI', 'VesselType']].drop_duplicates(),
            on='MMSI',
            how='inner'
        )
        
        logger.info(f"Données chargées: {len(merged_df)} échantillons")
        logger.info(f"Distribution des classes:\n{merged_df['VesselType'].value_counts()}")
        
        # Convertir VesselType en catégories
        merged_df['VesselCategory'] = merged_df['VesselType'].apply(self.vessel_type_to_category)
        
        logger.info(f"Distribution des catégories:\n{merged_df['VesselCategory'].value_counts()}")
        
        # Filtrer les classes avec moins de 2 échantillons pour la validation croisée
        class_counts = merged_df['VesselCategory'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        
        if len(valid_classes) < len(class_counts):
            removed_classes = class_counts[class_counts < 2].index
            logger.warning(f"Suppression des catégories avec <2 échantillons: {removed_classes.tolist()}")
            merged_df = merged_df[merged_df['VesselCategory'].isin(valid_classes)]
            logger.info(f"Données après filtrage: {len(merged_df)} échantillons")
            logger.info(f"Nouvelles catégories: {merged_df['VesselCategory'].value_counts()}")
        
        # Préparer X et y
        X = merged_df[self.feature_columns]
        y = merged_df['VesselCategory']
        
        # Nettoyer les données
        X_clean = self.clean_data(X)
        
        return X_clean, y
    
    def check_multicollinearity(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la multicolinéarité avec le Variance Inflation Factor (VIF).
        
        Args:
            X: DataFrame des features
            
        Returns:
            pd.DataFrame: DataFrame avec les scores VIF
        """
        logger.info("Vérification de la multicolinéarité (VIF)...")
        
        try:
            # Vérifier qu'il n'y a pas de valeurs infinies ou NaN
            if np.isinf(X.values).any() or X.isnull().any().any():
                logger.error("Les données contiennent encore des valeurs infinies ou NaN")
                return pd.DataFrame()
            
            # Ajouter une constante pour éviter la singularité
            X_with_const = X.copy()
            X_with_const['const'] = 1
            
            # Standardiser les données pour le calcul VIF
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_with_const)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_with_const.columns)
            
            # Vérifier les données standardisées
            if np.isinf(X_scaled).any() or np.isnan(X_scaled).any():
                logger.error("Les données standardisées contiennent des valeurs infinies ou NaN")
                return pd.DataFrame()
            
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns  # Exclure la constante
            vif_scores = []
            
            for i in range(len(X.columns)):  # Exclure la constante
                try:
                    vif_score = variance_inflation_factor(X_scaled_df.values, i)
                    # Vérifier si le score VIF est valide
                    if np.isfinite(vif_score):
                        vif_scores.append(vif_score)
                    else:
                        vif_scores.append(np.nan)
                        logger.warning(f"VIF non calculable pour {X.columns[i]}")
                except Exception as e:
                    logger.warning(f"Erreur VIF pour {X.columns[i]}: {str(e)}")
                    vif_scores.append(np.nan)
            
            vif_data["VIF"] = vif_scores
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            # Afficher seulement les VIF valides
            valid_vif = vif_data.dropna()
            if not valid_vif.empty:
                logger.info(f"VIF scores:\n{valid_vif}")
                
                # Signaler les features avec VIF élevé (>10)
                high_vif = valid_vif[valid_vif['VIF'] > 10]
                if not high_vif.empty:
                    logger.warning(f"Features avec VIF élevé (>10):\n{high_vif}")
            else:
                logger.warning("Aucun score VIF valide calculé")
            
            return vif_data
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul VIF: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, fit_transformers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prétraite les données (normalisation, encodage des labels).
        
        Args:
            X: Features
            y: Labels
            fit_transformers: Si True, fit les transformateurs, sinon utilise les existants
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features et labels transformés
        """
        logger.info("Prétraitement des données...")
        
        if fit_transformers:
            # Normalisation robuste (moins sensible aux outliers)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Encodage des labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            logger.info(f"Classes encodées: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        else:
            X_scaled = self.scaler.transform(X)
            y_encoded = self.label_encoder.transform(y)
        
        return X_scaled, y_encoded
    
    def create_models(self, n_classes: int) -> Dict[str, Any]:
        """
        Crée les différents modèles à tester.
        
        Args:
            n_classes: Nombre de classes
            
        Returns:
            Dict[str, Any]: Dictionnaire des modèles
        """
        logger.info("Création des modèles...")
        
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced',
                    verbosity=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            },
            
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            },
        }
        
        return models
    
    def train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Entraîne et évalue tous les modèles avec GridSearchCV.
        
        Args:
            X: Features normalisées
            y: Labels encodés
            
        Returns:
            Dict[str, Dict]: Résultats d'évaluation pour chaque modèle
        """
        logger.info("Entraînement et évaluation des modèles...")
        
        # Vérifier le nombre minimum d'échantillons par classe
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        logger.info(f"Nombre minimum d'échantillons par classe: {min_class_count}")
        
        # Ajuster le nombre de folds si nécessaire
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            logger.error("Pas assez d'échantillons pour la validation croisée")
            raise ValueError("Au moins 2 échantillons par classe sont nécessaires")
        
        logger.info(f"Utilisation de {n_splits} folds pour la validation croisée")
        
        # Split train/test avec stratification
        test_size = max(0.1, min(0.3, 1.0 / min_class_count))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # SMOTE pour équilibrer les classes sur le train set uniquement
        unique_train, train_counts = np.unique(y_train, return_counts=True)
        min_train_count = np.min(train_counts)
        
        if min_train_count >= 2:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_train_count-1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"Après SMOTE - Distribution: {np.bincount(y_train_balanced)}")
            except Exception as e:
                logger.warning(f"SMOTE échoué: {str(e)} - Utilisation des données non équilibrées")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            logger.warning("SMOTE non applicable - pas assez d'échantillons par classe")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        models = self.create_models(len(np.unique(y)))
        results = {}
        
        # Cross-validation stratifiée adaptée
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for name, model_config in models.items():
            logger.info(f"Entraînement du modèle: {name}")
            
            try:
                # Adapter les paramètres pour les petits datasets
                adapted_params = self.adapt_params_for_small_dataset(
                    model_config['params'], len(X_train_balanced)
                )
                
                # GridSearchCV avec gestion d'erreurs améliorée
                grid_search = GridSearchCV(
                    model_config['model'],
                    adapted_params,
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=1,
                    verbose=0,
                    error_score='raise'  # Pour déboguer les erreurs
                )
                
                # Fit sur les données équilibrées
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                # Meilleur modèle
                best_model = grid_search.best_estimator_
                self.models[name] = best_model
                
                # Prédictions sur le test set
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)
                
                # Métriques
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                
                # ROC AUC pour classification multi-classe
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except Exception as e:
                    logger.warning(f"Impossible de calculer ROC AUC pour {name}: {str(e)}")
                    roc_auc = None
                
                results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'classification_report': classification_report(y_test, y_pred, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                logger.info(f"{name} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
                
                # Feature importance si disponible
                if hasattr(best_model, 'feature_importances_'):
                    self.feature_importance[name] = best_model.feature_importances_
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.evaluation_results = results
        
        # Sélection du meilleur modèle basé sur le F1 score
        best_f1 = 0
        for name, result in results.items():
            if 'f1_score' in result and result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        if self.best_model_name:
            logger.info(f"Meilleur modèle: {self.best_model_name} (F1: {best_f1:.4f})")
        else:
            logger.warning("Aucun modèle valide trouvé")
        
        return results
    
    def adapt_params_for_small_dataset(self, params: Dict, n_samples: int) -> Dict:
        """
        Adapte les hyperparamètres pour les petits datasets.
        
        Args:
            params: Paramètres originaux
            n_samples: Nombre d'échantillons
            
        Returns:
            Dict: Paramètres adaptés
        """
        adapted_params = params.copy()
        
        # Réduire les paramètres pour éviter l'overfitting sur petits datasets
        if n_samples < 200:
            # Réduire n_estimators
            if 'n_estimators' in adapted_params:
                adapted_params['n_estimators'] = [50, 100, 150]
            
            # Réduire max_depth
            if 'max_depth' in adapted_params:
                adapted_params['max_depth'] = [3, 5, 10]
            
            # Réduire iterations pour CatBoost
            if 'iterations' in adapted_params:
                adapted_params['iterations'] = [50, 100, 150]
                
            # Réduire depth pour CatBoost
            if 'depth' in adapted_params:
                adapted_params['depth'] = [3, 4, 6]
                
            # Réduire num_leaves pour LightGBM
            if 'num_leaves' in adapted_params:
                adapted_params['num_leaves'] = [15, 31, 50]
        
        return adapted_params
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle et les transformateurs."""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'evaluation_results': self.evaluation_results
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.evaluation_results = model_data.get('evaluation_results', {})
        
        logger.info(f"Modèle chargé: {filepath}")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Prédit le type de navire pour de nouvelles features.
        
        Args:
            features: Array des features (1D ou 2D)
            
        Returns:
            Tuple[str, float]: Classe prédite et probabilité maximale
        """
        if self.best_model is None:
            raise ValueError("Aucun modèle entraîné. Utilisez train() d'abord.")
        
        # Reshaper si nécessaire
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normaliser
        features_scaled = self.scaler.transform(features)
        
        # Prédire
        prediction = self.best_model.predict(features_scaled)
        probabilities = self.best_model.predict_proba(features_scaled)
        
        # Décoder le label
        vessel_type = self.label_encoder.inverse_transform(prediction)[0]
        max_probability = np.max(probabilities[0])
        
        return vessel_type, max_probability
    
    def predict_from_mmsi(self, mmsi: int, features_file: str) -> Tuple[str, float]:
        """
        Prédit le type de navire à partir d'un MMSI.
        
        Args:
            mmsi: MMSI du navire
            features_file: Fichier des features
            
        Returns:
            Tuple[str, float]: Classe prédite et probabilité
        """
        features_df = pd.read_csv(features_file)
        
        # Trouver le navire
        vessel_data = features_df[features_df['MMSI'] == mmsi]
        
        if vessel_data.empty:
            raise ValueError(f"MMSI {mmsi} non trouvé dans les données")
        
        # Extraire les features
        features = vessel_data[self.feature_columns].values
        
        # Nettoyer les features
        features_df_clean = self.clean_data(pd.DataFrame(features, columns=self.feature_columns))
        features_clean = features_df_clean.values
        
        return self.predict(features_clean)
    
    def print_evaluation_summary(self):
        """Affiche un résumé de l'évaluation de tous les modèles."""
        if not self.evaluation_results:
            logger.warning("Aucun résultat d'évaluation disponible")
            return
        
        print("\n" + "="*70)
        print("RÉSUMÉ DE L'ÉVALUATION DES MODÈLES")
        print("="*70)
        
        for name, results in self.evaluation_results.items():
            if 'error' in results:
                print(f"\n{name}: ERREUR - {results['error']}")
                continue
                
            print(f"\n{name}:")
            print(f"  Meilleurs paramètres: {results['best_params']}")
            print(f"  Score CV: {results['best_score']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1 Score: {results['f1_score']:.4f}")
            if results['roc_auc']:
                print(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        print(f"\n🏆 MEILLEUR MODÈLE: {self.best_model_name}")
        print("="*70)
        
        # Afficher la classification détaillée du meilleur modèle
        if self.best_model_name and self.best_model_name in self.evaluation_results:
            print(f"\nRapport de classification détaillé - {self.best_model_name}:")
            print(self.evaluation_results[self.best_model_name]['classification_report'])

def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la prédiction du type de navire"
    )
    
    # Arguments principaux
    parser.add_argument('--train', action='store_true', 
                       help='Entraîner les modèles')
    parser.add_argument('--evaluate', action='store_true',
                       help='Évaluer les modèles')
    parser.add_argument('--predict', action='store_true',
                       help='Faire une prédiction')
    
    # Fichiers de données
    parser.add_argument('--features-file', default='../data/features_completes.csv',
                       help='Fichier des features (défaut: ../data/features_completes.csv)')
    parser.add_argument('--raw-data-file', default='../data/export_IA_final.csv',
                       help='Fichier des données brutes (../data/export_IA_final.csv)')
    parser.add_argument('--model-file', default='../data//vessel_model.pkl',
                       help='Fichier du modèle (défaut: ../data/vessel_model.pkl)')
    
    # Options de prédiction
    parser.add_argument('--mmsi', type=int,
                       help='MMSI pour la prédiction')
    parser.add_argument('--features', type=str,
                       help='Features séparées par des virgules pour la prédiction')
    
    args = parser.parse_args()
    
    # Initialiser le prédicteur
    predictor = VesselTypePredictor()
    
    try:
        if args.train or args.evaluate:
            # Charger et préparer les données
            X, y = predictor.load_and_prepare_data(args.features_file, args.raw_data_file)
            
            # Vérifier la multicolinéarité
            vif_scores = predictor.check_multicollinearity(X)
            
            # Prétraitement
            X_processed, y_processed = predictor.preprocess_data(X, y)
            
            # Entraînement et évaluation
            results = predictor.train_and_evaluate_models(X_processed, y_processed)
            
            # Sauvegarder le modèle
            predictor.save_model(args.model_file)
            
            if args.evaluate:
                predictor.print_evaluation_summary()
        
        elif args.predict:
            # Charger le modèle
            if Path(args.model_file).exists():
                predictor.load_model(args.model_file)
            else:
                raise FileNotFoundError(f"Modèle non trouvé: {args.model_file}")
            
            if args.mmsi:
                # Prédiction à partir d'un MMSI
                vessel_type, probability = predictor.predict_from_mmsi(
                    args.mmsi, args.features_file
                )
                print(f"\nPrédiction pour MMSI {args.mmsi}:")
                print(f"Type de navire: {vessel_type}")
                print(f"Probabilité: {probability:.4f}")
                
            elif args.features:
                # Prédiction à partir de features
                features_list = [float(x.strip()) for x in args.features.split(',')]
                
                if len(features_list) != len(predictor.feature_columns):
                    raise ValueError(f"Nombre de features incorrect. Attendu: {len(predictor.feature_columns)}, Reçu: {len(features_list)}")
                
                features_array = np.array(features_list)
                vessel_type, probability = predictor.predict(features_array)
                
                print(f"\nPrédiction:")
                print(f"Type de navire: {vessel_type}")
                print(f"Probabilité: {probability:.4f}")
            
            else:
                print("Erreur: Spécifiez --mmsi ou --features pour la prédiction")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
