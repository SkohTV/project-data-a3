import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import sys
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')

"""
TUTOS POUR LA PRÉDICTION DU TYPE DE NAVIRE :
-> python vessel_prediction.py --train --evaluate --data ton_fichier.csv
-> python vessel_prediction.py --predict --mmsi 123456789 --data ton_fichier.csv
-> python vessel_prediction.py --predict --features "12.5,45.2,180,50,15,3.5,5,A" --data ton_fichier.csv

Format des features brutes pour --predict --features :
SOG,COG,Heading,Length,Width,Draft,Status,TransceiverClass
Exemple: "12.5,45.2,180,50,15,3.5,5,A"
Status: entier entre 0 et 15
TransceiverClass: 'A' ou 'B'
"""

RANDOM_STATE = 42

class VesselTypePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {} 
        self.models = {}
        self.best_models = {}
        self.feature_names = []
        self.feature_info = {}
        self.raw_feature_names = ['SOG', 'COG', 'Heading', 'Length', 'Width', 'Draft', 'Status', 'TransceiverClass']
        
    def load_and_prepare_data(self, file_path):
        """
        CHARGER LES DONNÉES ET PRÉPARER LE DATAFRAME
        file_path: CHEMIN VERS LE FICHIER CSV DONT LES DONNÉES SERONT CHARGÉES
        """
        # print("-> CHARGEMENT DES DONNÉES...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        df = pd.read_csv(file_path)
        
        df = df[(df['VesselType'] >= 60) & (df['VesselType'] <= 89)]
        
        def categorize_vessel_type(vessel_type):
            if 60 <= vessel_type <= 69:
                return 'Passenger'
            elif 70 <= vessel_type <= 79:
                return 'Cargo'
            elif 80 <= vessel_type <= 89:
                return 'Tanker'
            else:
                return 'Unknown'
        
        df['VesselCategory'] = df['VesselType'].apply(categorize_vessel_type)
        
        return df
    
    def validate_raw_features(self, raw_features):
        """
        VALIDER LES FEATURES BRUTES SELON LES NOUVEAUX CRITÈRES
        """
        if len(raw_features) != 8:
            raise ValueError(f"Attendu 8 features, reçu {len(raw_features)}")
        
        sog, cog, heading, length, width, draft, status, transceiver_class = raw_features
        
        # Validation du Status (doit être un entier entre 0 et 15)
        try:
            status_int = int(status)
            if not (0 <= status_int <= 15):
                raise ValueError(f"Status doit être un entier entre 0 et 15, reçu: {status}")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Status doit être un entier, reçu: {status}")
            else:
                raise e
        
        # Validation du TransceiverClass (doit être 'A' ou 'B')
        if str(transceiver_class).upper() not in ['A', 'B']:
            raise ValueError(f"TransceiverClass doit être 'A' ou 'B', reçu: {transceiver_class}")
        
        # Validation des features numériques
        numeric_features = [sog, cog, heading, length, width, draft]
        for i, feature in enumerate(numeric_features):
            try:
                float(feature)
            except ValueError:
                feature_names = ['SOG', 'COG', 'Heading', 'Length', 'Width', 'Draft']
                raise ValueError(f"{feature_names[i]} doit être un nombre, reçu: {feature}")
        
        return True
    
    def process_raw_features_to_engineered(self, raw_features):
        """
        CONVERTIR DES FEATURES BRUTES EN FEATURES ENGINEERÉES
        raw_features: liste des valeurs brutes [SOG, COG, Heading, Length, Width, Draft, Status, TransceiverClass]
        """
        # print("-> CONVERSION DES FEATURES BRUTES EN FEATURES ENGINEERÉES...")
        
        # Validation des features
        self.validate_raw_features(raw_features)
        
        # Mapping des features brutes
        sog, cog, heading, length, width, draft, status, transceiver_class = raw_features
        
        # Créer un dictionnaire pour les features
        engineered = {}
        
        # Features numériques directes
        engineered['SOG'] = float(sog) if sog != '' else 0.0
        engineered['COG'] = float(cog) if cog != '' else 0.0
        engineered['Heading'] = float(heading) if heading != '' else 0.0
        engineered['Length'] = float(length) if length != '' else 0.0
        engineered['Width'] = float(width) if width != '' else 0.0
        engineered['Draft'] = float(draft) if draft != '' else 0.0
        
        # Features temporelles (valeurs par défaut car on n'a pas de BaseDateTime)
        engineered['Hour'] = 12  # Midi par défaut
        engineered['DayOfWeek'] = 1  # Lundi par défaut  
        engineered['Month'] = 6  # Juin par défaut
        
        # Zones géographiques (valeurs par défaut car on n'a pas LAT/LON)
        engineered['LAT_Zone'] = 5  # Zone centrale
        engineered['LON_Zone'] = 5  # Zone centrale
        
        # Ratio longueur/largeur
        if engineered['Width'] > 0 and engineered['Length'] > 0:
            engineered['Length_Width_Ratio'] = engineered['Length'] / engineered['Width']
        else:
            engineered['Length_Width_Ratio'] = 0.0
        
        # Catégorie de vitesse
        speed_category = 'Unknown'
        if engineered['SOG'] == 0:
            speed_category = 'Stationary'
        elif 0 < engineered['SOG'] <= 5:
            speed_category = 'Slow'
        elif 5 < engineered['SOG'] <= 15:
            speed_category = 'Medium'
        elif engineered['SOG'] > 15:
            speed_category = 'Fast'
        
        # Catégorie de taille
        size_category = 'Unknown'
        if 0 < engineered['Length'] <= 50:
            size_category = 'Small'
        elif 50 < engineered['Length'] <= 100:
            size_category = 'Medium'
        elif 100 < engineered['Length'] <= 200:
            size_category = 'Large'
        elif engineered['Length'] > 200:
            size_category = 'Very_Large'
        
        # Normalisation des valeurs d'entrée
        status_int = int(status)  # Déjà validé dans validate_raw_features
        transceiver_normalized = str(transceiver_class).upper()  # 'A' ou 'B'
        
        # Encodage des features catégorielles
        categorical_features = {
            'Speed_Category': speed_category,
            'Size_Category': size_category,
            'Status': str(status_int),  # Convertir en string pour l'encodage
            'TransceiverClass': transceiver_normalized
        }
        
        for feature, value in categorical_features.items():
            if feature in self.label_encoders:
                try:
                    # Essayer d'encoder avec les classes connues
                    encoded_value = self.label_encoders[feature].transform([value])[0]
                except ValueError:
                    # Si la valeur n'est pas connue, utiliser la classe la plus fréquente (0)
                    # print(f"-> WARNING: Valeur '{value}' inconnue pour {feature}, utilisation de la valeur par défaut")
                    encoded_value = 0
            else:
                # Si l'encodeur n'existe pas, utiliser 0
                encoded_value = 0
            
            engineered[feature + '_encoded'] = encoded_value
        
        return engineered
    
    def feature_engineering(self, df):
        """
        CRÉER DES FEATURES À PARTIR DU DATAFRAME
        df: DATAFRAME AVEC LES DONNÉES INITIALES
        """
        # print("\n-> CRÉATION DES FEATURES...")
        
        df_processed = df.copy()

        # Traitement des dates
        if 'BaseDateTime' in df_processed.columns:
            df_processed['BaseDateTime'] = pd.to_datetime(df_processed['BaseDateTime'], errors='coerce')
            df_processed['Hour'] = df_processed['BaseDateTime'].dt.hour
            df_processed['DayOfWeek'] = df_processed['BaseDateTime'].dt.dayofweek
            df_processed['Month'] = df_processed['BaseDateTime'].dt.month
        else:
            # print("WARNING: Colonne 'BaseDateTime' manquante")
            df_processed['Hour'] = 12
            df_processed['DayOfWeek'] = 1
            df_processed['Month'] = 6
        
        numeric_base_features = ['SOG', 'COG', 'Heading', 'Length', 'Width', 'Draft']
        
        for col in numeric_base_features:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                # print(f"Colonne {col}: {df_processed[col].dtype}, NaN: {df_processed[col].isna().sum()}")
        
        # Catégories de vitesse
        if 'SOG' in df_processed.columns:
            df_processed['Speed_Category'] = pd.cut(df_processed['SOG'], 
                                                  bins=[-1, 0, 5, 15, float('inf')], 
                                                  labels=['Stationary', 'Slow', 'Medium', 'Fast'])
        else:
            df_processed['Speed_Category'] = 'Unknown'
        
        # Zones géographiques
        if 'LAT' in df_processed.columns and 'LON' in df_processed.columns:
            df_processed['LAT'] = pd.to_numeric(df_processed['LAT'], errors='coerce')
            df_processed['LON'] = pd.to_numeric(df_processed['LON'], errors='coerce')
            df_processed['LAT_Zone'] = pd.cut(df_processed['LAT'], bins=10, labels=False)
            df_processed['LON_Zone'] = pd.cut(df_processed['LON'], bins=10, labels=False)
        else:
            df_processed['LAT_Zone'] = 5
            df_processed['LON_Zone'] = 5
        
        # Catégories de taille
        if 'Length' in df_processed.columns:
            df_processed['Size_Category'] = pd.cut(df_processed['Length'], 
                                                 bins=[0, 50, 100, 200, float('inf')], 
                                                 labels=['Small', 'Medium', 'Large', 'Very_Large'])
        else:
            df_processed['Size_Category'] = 'Unknown'
        
        # Ratio longueur/largeur
        if ('Length' in df_processed.columns and 'Width' in df_processed.columns):
            df_processed['Length_Width_Ratio'] = np.where(
                (df_processed['Width'] > 0) & (df_processed['Length'].notna()) & (df_processed['Width'].notna()), 
                df_processed['Length'] / df_processed['Width'], 
                0
            )
        else:
            df_processed['Length_Width_Ratio'] = 0
        
        # Normalisation des données catégorielles pour l'entraînement
        if 'Status' in df_processed.columns:
            # Conversion du Status en entier si possible, sinon utiliser une valeur par défaut
            df_processed['Status'] = pd.to_numeric(df_processed['Status'], errors='coerce')
            df_processed['Status'] = df_processed['Status'].fillna(0).astype(int)
            # Clipper les valeurs entre 0 et 15
            df_processed['Status'] = np.clip(df_processed['Status'], 0, 15)
            df_processed['Status'] = df_processed['Status'].astype(str)  # Convertir en string pour l'encodage
        
        if 'TransceiverClass' in df_processed.columns:
            # Normaliser TransceiverClass en 'A' ou 'B'
            df_processed['TransceiverClass'] = df_processed['TransceiverClass'].astype(str)
            df_processed['TransceiverClass'] = df_processed['TransceiverClass'].str.upper()
            # Remplacer les valeurs non-valides par 'A' par défaut
            valid_classes = ['A', 'B']
            df_processed['TransceiverClass'] = df_processed['TransceiverClass'].apply(
                lambda x: x if x in valid_classes else 'A'
            )
        
        # Encodage des features catégorielles
        categorical_features = ['Speed_Category', 'Size_Category', 'Status', 'TransceiverClass']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].astype(str).fillna('Unknown')
                
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                
                try:
                    df_processed[feature + '_encoded'] = self.label_encoders[feature].fit_transform(df_processed[feature])
                except Exception as e:
                    # print(f"Erreur lors de l'encodage de {feature}: {e}")
                    df_processed[feature + '_encoded'] = 0
            else:
                df_processed[feature + '_encoded'] = 0
        
        # Sélection des features finales
        available_numeric_features = []
        for feature in numeric_base_features:
            if (feature in df_processed.columns and 
                df_processed[feature].dtype in ['float64', 'int64', 'float32', 'int32']):
                available_numeric_features.append(feature)
        
        categorical_encoded_features = [f + '_encoded' for f in categorical_features]
        engineered_features = ['Hour', 'DayOfWeek', 'Month', 'LAT_Zone', 'LON_Zone', 'Length_Width_Ratio']
        
        feature_columns = available_numeric_features + categorical_encoded_features + engineered_features
        
        # Nettoyage des données
        df_processed = df_processed.dropna(subset=['VesselCategory'])
        
        for col in feature_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                
                if df_processed[col].isna().any():
                    median_val = df_processed[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_processed[col] = df_processed[col].fillna(median_val)
                
                if df_processed[col].std() > 0:
                    q99 = df_processed[col].quantile(0.99)
                    q01 = df_processed[col].quantile(0.01)
                    df_processed[col] = np.clip(df_processed[col], q01, q99)
            else:
                df_processed[col] = 0

        # Sauvegarde des infos features
        self.feature_names = feature_columns
        self.feature_info = {}
        for col in feature_columns:
            if col in df_processed.columns and len(df_processed[col]) > 0:
                self.feature_info[col] = {
                    'dtype': str(df_processed[col].dtype),
                    'min': float(df_processed[col].min()),
                    'max': float(df_processed[col].max()),
                    'mean': float(df_processed[col].mean())
                }
            else:
                self.feature_info[col] = {
                    'dtype': 'float64',
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0
                }
        
        return df_processed, feature_columns
    
    def split_data_by_vessel(self, df, feature_columns, test_size=0.2):
        """SÉPARATION DES DONNÉES PAR NAVIRE POUR ÉVITER LE DATA LEAKAGE"""
        # print(f"\n-> SÉPARATION DES DONNÉES PAR NAVIRE (test_size={test_size})...")
        
        unique_vessels = df['MMSI'].unique()
        
        vessels_train, vessels_test = train_test_split(unique_vessels, test_size=test_size, random_state=RANDOM_STATE)
        
        train_df = df[df['MMSI'].isin(vessels_train)]
        test_df = df[df['MMSI'].isin(vessels_test)]
        
        # print(f"Navires dans le train: {len(vessels_train)}")
        # print(f"Navires dans le test: {len(vessels_test)}")
        # print(f"Points de données train: {len(train_df)}")
        # print(f"Points de données test: {len(test_df)}")

        X_train = train_df[feature_columns].copy()
        y_train = train_df['VesselCategory'].copy()
        X_test = test_df[feature_columns].copy()
        y_test = test_df['VesselCategory'].copy()
        
        for col in feature_columns:
            if col in X_train.columns:
                if X_train[col].dtype == 'object':
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """NORMALISATION DES FEATURES AVEC STANDARD SCALER"""
        # print("\n-> NORMALISATION DES FEATURES...")
        
        X_train_array = np.array(X_train, dtype=float)
        X_test_array = np.array(X_test, dtype=float)
        
        X_train_clean = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_clean = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.isfinite(X_train_clean).all():
            ...
            # print("   -> IL Y A DES VALEURS NON-FINIES DANS X_train")
        if not np.isfinite(X_test_clean).all():
            ...
            # print("   -> IL Y A DES VALEURS NON-FINIES DANS X_test")
        
        try:
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            X_test_scaled = self.scaler.transform(X_test_clean)
            # print("NORMALISATION OK")
        except Exception as e:
            # print(f"   ERREUR LORS DE LA NORMALISATION: {e}")
            X_train_scaled = X_train_clean
            X_test_scaled = X_test_clean
            # print("   -> UTILISATION DES DONNÉES NON NORMALISÉES EN RAISON D'UNE ERREUR")
        
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train):
        """ENTRAÎNER LES MODÈLES AVEC GRIDSEARCHCV POUR OPTIMISER LES HYPERPARAMÈTRES"""
        # print("\n-> ENTRAÎNEMENT DES MODÈLES...")
        
        models_params = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            }
        }
        
        for name, config in models_params.items():
            # print(f"\n-> ENTRAÎNEMENT DU MODÈLE: {name}")
            
            try:
                grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=1, verbose=0)
                
                grid_search.fit(X_train, y_train)
                
                self.best_models[name] = grid_search.best_estimator_
                # print(f"-> MEILLEUR PARAMÈTRES POUR {name}: {grid_search.best_params_}")
                # print(f"-> SCORE DE VALIDATION CROISÉE: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                # print(f"-> ERREUR LORS DE L'ENTRAÎNEMENT DU MODÈLE {name}: {e}")
                self.best_models[name] = config['model']
                self.best_models[name].fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """ÉVALUATION DES MODÈLES EN UTILISANT LES DONNÉES DE TEST"""
        # print("\n-> ÉVALUATION DES MODÈLES...")
        
        results = {}
        
        for name, model in self.best_models.items():
            # print(f"\n--- {name} ---")
            
            try:
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'model': model
                }
                
                # print(f"-> ACCURACY: {accuracy:.4f}")
                # print("\nRAPPORT DE CLASSIFICATION:")
                # print(classification_report(y_test, y_pred))
                
            except Exception as e:
                # print(f"-> ERREUR LORS DE L'ÉVALUATION DU MODÈLE {name}: {e}")
                ...
        
        return results
    
    def plot_results(self, y_test, results):
        """VISUALISATION DES RÉSULTATS DES MODÈLES"""
            
        # print("\n-> GÉNÉRATION DES GRAPHIQUES...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Évaluation des Modèles de Prédiction du Type de Navire', fontsize=16)
        
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Comparaison des Accuracies')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        vessel_counts = pd.Series(y_test).value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axes[0, 1].pie(vessel_counts.values, labels=vessel_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('Distribution des Catégories de Navires (Test Set)')
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Matrice de Confusion - {best_model_name}')
        axes[1, 0].set_xlabel('Prédictions')
        axes[1, 0].set_ylabel('Vraies Valeurs')
        
        correct_predictions = {}
        for name, result in results.items():
            correct_predictions[name] = np.sum(result['predictions'] == y_test)
        
        models = list(correct_predictions.keys())
        correct_counts = list(correct_predictions.values())
        
        axes[1, 1].bar(models, correct_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title('Nombre de Prédictions Correctes')
        axes[1, 1].set_ylabel('Nombre de Prédictions Correctes')
        for i, count in enumerate(correct_counts):
            axes[1, 1].text(i, count + 1, str(count), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name
    
    def save_model(self, best_model_name, file_path='./models/best_vessel_model.pkl'):
        """SAUVEGARDER LE MEILLEUR MODÈLE AVEC TOUTES LES INFORMATIONS NÉCESSAIRES"""
        
        # Créer le dossier models s'il n'existe pas
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_data = {
            'model': self.best_models[best_model_name],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'model_name': best_model_name,
            'feature_names': self.feature_names,
            'feature_info': self.feature_info,
            'raw_feature_names': self.raw_feature_names
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # print(f"\n-> MODÈLE SAUVEGARDÉ SOUS {file_path}")
        # print(f"-> MEILLEUR MODÈLE: {best_model_name}")
        # print(f"-> FEATURES SAUVEGARDÉES: {len(self.feature_names)}")

    def show_feature_info_from_model(self, model_path):
        """AFFICHER LES INFORMATIONS SUR LES FEATURES À PARTIR DU MODÈLE SAUVEGARDÉ"""
        # print("-> ANALYSE DES FEATURES À PARTIR DU MODÈLE SAUVEGARDÉ...")
        
        if not self.load_model(model_path):
            # print("-> IMPOSSIBLE DE CHARGER LE MODÈLE!")
            return None
        
        # print(f"\n-> FORMAT DES FEATURES BRUTES POUR LA PRÉDICTION:")
        # print("Les features doivent être fournies dans l'ordre suivant:")
        feature_descriptions = [
            "SOG (Speed Over Ground) - nombre décimal",
            "COG (Course Over Ground) - nombre décimal", 
            "Heading - nombre décimal",
            "Length - nombre décimal",
            "Width - nombre décimal", 
            "Draft - nombre décimal",
            "Status - entier entre 0 et 15",
            "TransceiverClass - 'A' ou 'B'"
        ]
        for i, desc in enumerate(feature_descriptions):
            ...
            # print(f"  {i+1}. {desc}")
        
        # print(f"\n-> EXEMPLE DE COMMANDE:")
        # print("python vessel_prediction.py --predict --features \"12.5,45.2,180,50,15,3.5,5,A\"")
        
        # print(f"\n-> FEATURES ENGINEERÉES UTILISÉES DANS LE MODÈLE ({len(self.feature_names)} features):")
        for i, feature in enumerate(self.feature_names):
            info = self.feature_info.get(feature, {'min': 0, 'max': 0, 'mean': 0})
            # print(f"  {i+1:2d}. {feature:25s} - Min: {info['min']:.2f}, Max: {info['max']:.2f}, Moyenne: {info['mean']:.2f}")
        
        return self.feature_names

    def show_feature_info(self, file_path):
        """AFFICHER LES INFORMATIONS SUR LES FEATURES UTILISÉES"""
        # print("-> ANALYSE DES FEATURES UTILISÉES...")
        
        try:
            df = self.load_and_prepare_data(file_path)
            df_processed, features = self.feature_engineering(df)
            
            # print(f"\n-> FORMAT DES FEATURES BRUTES POUR LA PRÉDICTION:")
            # print("Les features doivent être fournies dans l'ordre suivant:")
            feature_descriptions = [
                "SOG (Speed Over Ground) - nombre décimal",
                "COG (Course Over Ground) - nombre décimal", 
                "Heading - nombre décimal",
                "Length - nombre décimal",
                "Width - nombre décimal", 
                "Draft - nombre décimal",
                "Status - entier entre 0 et 15",
                "TransceiverClass - 'A' ou 'B'"
            ]
            for i, desc in enumerate(feature_descriptions):
                ...
                # print(f"  {i+1}. {desc}")
            
            # print(f"\n-> EXEMPLE DE COMMANDE:")
            # print("python vessel_prediction.py --predict --features \"12.5,45.2,180,50,15,3.5,5,A\"")
            
            # print(f"\n-> FEATURES ENGINEERÉES UTILISÉES ({len(features)} features):")
            for i, feature in enumerate(features):
                if feature in df_processed.columns:
                    col_data = df_processed[feature]
                    # print(f"  {i+1:2d}. {feature:25s} - Min: {col_data.min():.2f}, Max: {col_data.max():.2f}, Moyenne: {col_data.mean():.2f}")
                else:
                    ...
                    # print(f"  {i+1:2d}. {feature:25s} - Feature non trouvée")
                    
        except Exception as e:
            # print(f"-> ERREUR LORS DE L'ANALYSE DES FEATURES: {e}")
            return None
        
        return features

    def load_model(self, model_path='./models/best_vessel_model.pkl'):
        """CHARGER UN MODÈLE SAUVEGARDÉ"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_models = {'loaded_model': model_data['model']}
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.feature_info = model_data['feature_info']
            
            if 'raw_feature_names' in model_data:
                self.raw_feature_names = model_data['raw_feature_names']
            
            # print(f"-> MODÈLE CHARGÉ AVEC SUCCÈS: {model_data['model_name']}")
            return True
            
        except Exception as e:
            # print(f"-> ERREUR LORS DU CHARGEMENT DU MODÈLE: {e}")
            return False

    def predict_single(self, features, model_path='models/best_vessel_model.pkl'):
        """PRÉDIRE LE TYPE DE NAVIRE POUR UN SEUL ÉCHANTILLON"""
        # print("-> PRÉDICTION POUR UN ÉCHANTILLON...")
        
        if not self.load_model(model_path):
            return None
        
        try:
            # Conversion des features brutes en features engineerées
            engineered_features = self.process_raw_features_to_engineered(features)
            
            # Créer le vecteur de features dans le bon ordre
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in engineered_features:
                    feature_vector.append(engineered_features[feature_name])
                else:
                    # Utiliser la valeur par défaut si la feature n'est pas disponible
                    default_value = self.feature_info.get(feature_name, {}).get('mean', 0.0)
                    feature_vector.append(default_value)
                    # print(f"-> WARNING: Feature '{feature_name}' manquante, utilisation de la valeur par défaut: {default_value}")
            
            # Normalisation
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Prédiction
            model = self.best_models['loaded_model']
            prediction = model.predict(feature_vector_scaled)[0]
            prediction_proba = model.predict_proba(feature_vector_scaled)[0]
            
            # Affichage des résultats
            # print(f"\n-> RÉSULTAT DE LA PRÉDICTION:")
            # print(f"   TYPE DE NAVIRE PRÉDIT: {prediction}")
            
            # print(f"\n-> PROBABILITÉS PAR CLASSE:")
            stats = {}
            for i, class_name in enumerate(model.classes_):
                stats[class_name] = prediction_proba
                ...
                # print(f"   {class_name}: {prediction_proba[i]:.4f} ({prediction_proba[i]*100:.2f}%)")
            print(max(stats, key=stats.get))
            
            return {
                'prediction': prediction,
                'probabilities': dict(zip(model.classes_, prediction_proba)),
                'confidence': max(prediction_proba)
            }
            
        except Exception as e:
            # print(f"-> ERREUR LORS DE LA PRÉDICTION: {e}")
            return None

    def predict_by_mmsi(self, mmsi, data_file):
        """PRÉDIRE LE TYPE DE NAVIRE EN UTILISANT LES DONNÉES D'UN MMSI SPÉCIFIQUE"""
        # print(f"-> PRÉDICTION POUR LE NAVIRE MMSI: {mmsi}")
        
        try:
            df = pd.read_csv(data_file)
            vessel_data = df[df['MMSI'] == mmsi]
            
            if vessel_data.empty:
                # print(f"-> AUCUNE DONNÉE TROUVÉE POUR LE MMSI: {mmsi}")
                return None
            
            # Prendre la première ligne de données pour ce navire
            sample = vessel_data.iloc[0]
            
            # Extraire les features brutes
            raw_features = []
            for feature_name in self.raw_feature_names:
                if feature_name in sample:
                    raw_features.append(sample[feature_name])
                else:
                    # print(f"-> WARNING: Feature '{feature_name}' manquante pour MMSI {mmsi}")
                    # Valeurs par défaut
                    if feature_name == 'Status':
                        raw_features.append(0)
                    elif feature_name == 'TransceiverClass':
                        raw_features.append('A')
                    else:
                        raw_features.append(0.0)
            
            # print(f"-> FEATURES EXTRAITES: {raw_features}")
            
            # Vérifier si on a le vrai type de navire pour comparaison
            if 'VesselType' in sample:
                true_vessel_type = sample['VesselType']
                # print(f"-> VRAI TYPE DE NAVIRE (VesselType): {true_vessel_type}")
                
                # Catégoriser le vrai type
                if 60 <= true_vessel_type <= 69:
                    true_category = 'Passenger'
                elif 70 <= true_vessel_type <= 79:
                    true_category = 'Cargo'
                elif 80 <= true_vessel_type <= 89:
                    true_category = 'Tanker'
                else:
                    true_category = 'Unknown'
                
                # print(f"-> VRAIE CATÉGORIE: {true_category}")
            
            # Faire la prédiction
            return self.predict_single(raw_features)
            
        except Exception as e:
            # print(f"-> ERREUR LORS DE LA PRÉDICTION PAR MMSI: {e}")
            return None


def main():
    """FONCTION PRINCIPALE POUR GÉRER LES ARGUMENTS DE LIGNE DE COMMANDE"""
    parser = argparse.ArgumentParser(description='Prédiction du type de navire')
    
    parser.add_argument('--train', action='store_true', help='Entraîner les modèles')
    parser.add_argument('--evaluate', action='store_true', help='Évaluer les modèles')
    parser.add_argument('--predict', action='store_true', help='Faire une prédiction')
    parser.add_argument('--data', type=str, help='Chemin vers le fichier de données CSV')
    parser.add_argument('--model', type=str, default='./models/best_vessel_model.pkl', help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--mmsi', type=int, help='MMSI du navire pour la prédiction')
    parser.add_argument('--features', type=str, help='Features brutes séparées par des virgules')
    parser.add_argument('--info', action='store_true', help='Afficher les informations sur les features')
    
    args = parser.parse_args()
    
    # Vérifier que --data est fourni quand c'est nécessaire
    data_required_cases = [
        args.train,
        args.evaluate, 
        (args.predict and args.mmsi),  # Prédiction par MMSI nécessite les données
        (args.info and not os.path.exists(args.model))  # Info sans modèle nécessite les données
    ]
    
    if any(data_required_cases) and not args.data:
        # print("-> ERREUR: --data est requis pour cette opération")
        if args.train or args.evaluate:
            ...
            # print("   L'entraînement/évaluation nécessite un fichier de données")
        elif args.predict and args.mmsi:
            ...
            # print("   La prédiction par MMSI nécessite un fichier de données pour chercher le navire")
        elif args.info:
            ...
            # print("   --info nécessite --data si le modèle n'existe pas encore")
        sys.exit(1)
    
    predictor = VesselTypePredictor()
    
    if args.info:
        if os.path.exists(args.model):
            predictor.show_feature_info_from_model(args.model)
        else:
            predictor.show_feature_info(args.data)
        return
    
    if args.train or args.evaluate:
        # print("=" * 60)
        # print("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES DE PRÉDICTION")
        # print("=" * 60)
        
        try:
            # Chargement et préparation des données
            df = predictor.load_and_prepare_data(args.data)
            # print(f"-> DONNÉES CHARGÉES: {len(df)} lignes")
            # print(f"-> DISTRIBUTION DES CATÉGORIES:")
            # print(df['VesselCategory'].value_counts())
            
            # Feature engineering
            df_processed, feature_columns = predictor.feature_engineering(df)
            # print(f"-> FEATURES CRÉÉES: {len(feature_columns)}")
            
            X_train, X_test, y_train, y_test = predictor.split_data_by_vessel(df_processed, feature_columns)
            
            X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
            
            if args.train:
                predictor.train_models(X_train_scaled, y_train)
            
            if args.evaluate:
                results = predictor.evaluate_models(X_test_scaled, y_test)
                
                if results:
                    best_model = predictor.plot_results(y_test, results)
                    
                    predictor.save_model(best_model, args.model)
                
        except Exception as e:
            # print(f"-> ERREUR LORS DE L'ENTRAÎNEMENT/ÉVALUATION: {e}")
            sys.exit(1)
    
    elif args.predict:
        # print("=" * 60)
        # print("PRÉDICTION DU TYPE DE NAVIRE")
        # print("=" * 60)
        
        if not os.path.exists(args.model):
            # print(f"-> MODÈLE NON TROUVÉ: {args.model}")
            # print("-> VEUILLEZ D'ABORD ENTRAÎNER UN MODÈLE AVEC --train --evaluate")
            sys.exit(1)
        
        if args.mmsi:
            # Prédiction par MMSI
            result = predictor.predict_by_mmsi(args.mmsi, args.data)
            
        elif args.features:
            # Prédiction par features
            try:
                features = args.features.split(',')
                if len(features) != 8:
                    # print(f"-> ERREUR: Attendu 8 features, reçu {len(features)}")
                    # print("-> FORMAT ATTENDU: SOG,COG,Heading,Length,Width,Draft,Status,TransceiverClass")
                    # print("-> EXEMPLE: \"12.5,45.2,180,50,15,3.5,5,A\"")
                    sys.exit(1)
                
                # Convertir les features (tout sauf les deux dernières)
                parsed_features = []
                for i, feature in enumerate(features):
                    if i < 6:  # Les 6 premières sont numériques
                        parsed_features.append(float(feature.strip()))
                    else:  # Status et TransceiverClass
                        parsed_features.append(feature.strip())
                
                result = predictor.predict_single(parsed_features, args.model)
                
            except ValueError as e:
                # print(f"-> ERREUR DE FORMAT DES FEATURES: {e}")
                # print("-> FORMAT ATTENDU: SOG,COG,Heading,Length,Width,Draft,Status,TransceiverClass")
                # print("-> EXEMPLE: \"12.5,45.2,180,50,15,3.5,5,A\"")
                sys.exit(1)
        
        else:
            # print("-> ERREUR: Veuillez spécifier --mmsi ou --features pour la prédiction")
            sys.exit(1)
        
        if result:
            ...
            # print(f"\n-> PRÉDICTION TERMINÉE AVEC SUCCÈS")
            # print(f"-> CONFIANCE: {result['confidence']:.4f}")
    
    else:
        # print("-> VEUILLEZ SPÉCIFIER --train, --evaluate, --predict ou --info")
        # print_help()


if __name__ == "__main__":
    main()
