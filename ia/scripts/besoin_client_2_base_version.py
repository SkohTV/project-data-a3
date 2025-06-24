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

warnings.filterwarnings('ignore')

"""
TUTOS POUR LA PRÉDICTION DU TYPE DE NAVIRE :
-> python vessel_prediction.py --train --evaluate --data ton_fichier.csv
-> python vessel_prediction.py --predict --mmsi 123456789 --data ton_fichier.csv
-> python vessel_prediction.py --predict --features "12.34,56.78,90.12,34.56,78.90" --data ton_fichier.csv
"""

RANDOM_STATE = 42

class VesselTypePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {} 
        self.models = {}
        self.best_models = {}
        
    def load_and_prepare_data(self, file_path):
        """
        CHARGER LES DONNÉES ET PRÉPARER LE DATAFRAME
        file_path: CHEMIN VERS LE FICHIER CSV DONT LES DONNÉES SERONT CHARGÉES
        """
        print("-> CHARGEMENT DES DONNÉES...")
        
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
    
    def feature_engineering(self, df):
        """
        CRÉER DES FEATURES À PARTIR DU DATAFRAME
        df: DATAFRAME AVEC LES DONNÉES INITIALES
        """
        print("\n-> CRÉATION DES FEATURES...")
        
        df_processed = df.copy()

        df_processed['BaseDateTime'] = pd.to_datetime(df_processed['BaseDateTime'])
        
        df_processed['Hour'] = df_processed['BaseDateTime'].dt.hour
        df_processed['DayOfWeek'] = df_processed['BaseDateTime'].dt.dayofweek
        df_processed['Month'] = df_processed['BaseDateTime'].dt.month
        
        numeric_base_features = ['SOG', 'COG', 'Heading', 'Length', 'Width', 'Draft']
        
        for col in numeric_base_features:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                print(f"Colonne {col}: {df_processed[col].dtype}, NaN: {df_processed[col].isna().sum()}")
        
        if 'SOG' in df_processed.columns and df_processed['SOG'].dtype in ['float64', 'int64']:
            df_processed['Speed_Category'] = pd.cut(df_processed['SOG'], bins=[-1, 0, 5, 15, float('inf')], labels=['Stationary', 'Slow', 'Medium', 'Fast'])
        else:
            df_processed['Speed_Category'] = 'Unknown'
        
        if 'LAT' in df_processed.columns and 'LON' in df_processed.columns:
            df_processed['LAT'] = pd.to_numeric(df_processed['LAT'], errors='coerce')
            df_processed['LON'] = pd.to_numeric(df_processed['LON'], errors='coerce')
            df_processed['LAT_Zone'] = pd.cut(df_processed['LAT'], bins=10, labels=False)
            df_processed['LON_Zone'] = pd.cut(df_processed['LON'], bins=10, labels=False)
        else:
            df_processed['LAT_Zone'] = 0
            df_processed['LON_Zone'] = 0
        
        if 'Length' in df_processed.columns and df_processed['Length'].dtype in ['float64', 'int64']:
            df_processed['Size_Category'] = pd.cut(df_processed['Length'], bins=[0, 50, 100, 200, float('inf')], labels=['Small', 'Medium', 'Large', 'Very_Large'])
        else:
            df_processed['Size_Category'] = 'Unknown'
        
        if ('Length' in df_processed.columns and 'Width' in df_processed.columns and
            df_processed['Length'].dtype in ['float64', 'int64'] and 
            df_processed['Width'].dtype in ['float64', 'int64']):
            df_processed['Length_Width_Ratio'] = np.where((df_processed['Width'] > 0) & (df_processed['Length'].notna()) & (df_processed['Width'].notna()), df_processed['Length'] / df_processed['Width'], 0)
        else:
            df_processed['Length_Width_Ratio'] = 0
        
        categorical_features = ['Speed_Category', 'Size_Category', 'Status', 'TransceiverClass']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].astype(str).fillna('Unknown')
                
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                
                df_processed[feature + '_encoded'] = self.label_encoders[feature].fit_transform(df_processed[feature])
            else:
                df_processed[feature + '_encoded'] = 0
        
        excluded_columns = ["id", "MMSI", "BaseDateTime", "LAT", "LON", "VesselName", "IMO", "CallSign", "Cargo"]
        
        available_numeric_features = []
        for feature in numeric_base_features:
            if (feature in df_processed.columns and 
                df_processed[feature].dtype in ['float64', 'int64', 'float32', 'int32']):
                available_numeric_features.append(feature)
        
        categorical_encoded_features = [f + '_encoded' for f in categorical_features]
        
        engineered_features = ['Hour', 'DayOfWeek', 'Month', 'LAT_Zone', 'LON_Zone', 'Length_Width_Ratio']
        
        feature_columns = available_numeric_features + categorical_encoded_features + engineered_features
                
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

        print(f"\n-> VÉRIFICATION DES FEATURES CRÉÉES...")
        for col in feature_columns:
            if col in df_processed.columns:
                print(f"  {col}: {df_processed[col].dtype}, NaN: {df_processed[col].isna().sum()}")
                
                # Vérifier s'il y a des valeurs non-numériques
                if df_processed[col].dtype == 'object':
                    print(f"    -> WARNING: {col} EST DE TYPE 'object' AVEC {df_processed[col].nunique()} VALEURS UNIQUES")
        
        return df_processed, feature_columns
    
    def split_data_by_vessel(self, df, feature_columns, test_size=0.2):
        """SÉPARATION DES DONNÉES PAR NAVIRE POUR ÉVITER LE DATA LEAKAGE"""
        print(f"\n-> SÉPARATION DES DONNÉES PAR NAVIRE (test_size={test_size})...")
        
        unique_vessels = df['MMSI'].unique()
        
        vessels_train, vessels_test = train_test_split(unique_vessels, test_size=test_size, random_state=RANDOM_STATE)
        
        train_df = df[df['MMSI'].isin(vessels_train)]
        test_df = df[df['MMSI'].isin(vessels_test)]
        
        print(f"Navires dans le train: {len(vessels_train)}")
        print(f"Navires dans le test: {len(vessels_test)}")
        print(f"Points de données train: {len(train_df)}")
        print(f"Points de données test: {len(test_df)}")

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
        print("\n-> NORMALISATION DES FEATURES...")
        
        X_train_array = np.array(X_train, dtype=float)
        X_test_array = np.array(X_test, dtype=float)
        
        X_train_clean = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_clean = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.isfinite(X_train_clean).all():
            print("   -> IL Y A DES VALEURS NON-FINIES DANS X_train")
        if not np.isfinite(X_test_clean).all():
            print("   -> IL Y A DES VALEURS NON-FINIES DANS X_test")
        
        try:
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            X_test_scaled = self.scaler.transform(X_test_clean)
            print("NORMALISATION OK")
        except Exception as e:
            print(f"   ERREUR LORS DE LA NORMALISATION: {e}")
            X_train_scaled = X_train_clean
            X_test_scaled = X_test_clean
            print("   -> UTILISATION DES DONNÉES NON NORMALISÉES EN RAISON D'UNE ERREUR")
        
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train):
        """ENTRAÎNER LES MODÈLES AVEC GRIDSEARCHCV POUR OPTIMISER LES HYPERPARAMÈTRES"""
        print("\n-> ENTRAÎNEMENT DES MODÈLES...")
        
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
            print(f"\n-> ENTRAÎNEMENT DU MODÈLE: {name}")
            
            try:
                grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=1, verbose=0)
                
                grid_search.fit(X_train, y_train)
                
                self.best_models[name] = grid_search.best_estimator_
                print(f"-> MEILLEUR PARAMÈTRES POUR {name}: {grid_search.best_params_}")
                print(f"-> SCORE DE VALIDATION CROISÉE: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"-> ERREUR LORS DE L'ENTRAÎNEMENT DU MODÈLE {name}: {e}")
                self.best_models[name] = config['model']
                self.best_models[name].fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """ÉVALUATION DES MODÈLES EN UTILISANT LES DONNÉES DE TEST"""
        print("\n-> ÉVALUATION DES MODÈLES...")
        
        results = {}
        
        for name, model in self.best_models.items():
            print(f"\n--- {name} ---")
            
            try:
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'model': model
                }
                
                print(f"-> ACCURACY: {accuracy:.4f}")
                print("\nRAPPORT DE CLASSIFICATION:")
                print(classification_report(y_test, y_pred))
                
            except Exception as e:
                print(f"-> ERREUR LORS DE L'ÉVALUATION DU MODÈLE {name}: {e}")
        
        return results
    
    def plot_results(self, y_test, results):
        """VISUALISATION DES RÉSULTATS DES MODÈLES"""
            
        print("\n-> GÉNÉRATION DES GRAPHIQUES...")
        
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
    
    def save_model(self, best_model_name, file_path='best_vessel_model.pkl'):
        """SAUVEGARDER LE MEILLEUR MODÈLE"""

        model_data = {
            'model': self.best_models[best_model_name],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'model_name': best_model_name
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n-> MODÈLE SAUVEGARDÉ SOUS {file_path}")
        print(f"-> MEILLEUR MODÈLE: {best_model_name}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning pour la Prédiction du Type de Navire")
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle')
    parser.add_argument('--evaluate', action='store_true', help='Évaluer le modèle')
    parser.add_argument('--predict', action='store_true', help='Prédire le type de navire')
    parser.add_argument('--mmsi', type=int, help='MMSI du navire à prédire')
    parser.add_argument('--features', type=str, help='Liste des features séparées par des virgules')
    parser.add_argument('--data', type=str, default='vessel_data.csv', help='Chemin vers le fichier CSV de données')

    args = parser.parse_args()
    predictor = VesselTypePredictor()

    if args.train or args.evaluate:
        df = predictor.load_and_prepare_data(args.data)
        df_processed, features = predictor.feature_engineering(df)
        X_train, X_test, y_train, y_test = predictor.split_data_by_vessel(df_processed, features)
        X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
        
        if args.train:
            predictor.train_models(X_train_scaled, y_train)
        if args.evaluate:
            results = predictor.evaluate_models(X_test_scaled, y_test)
            predictor.plot_results(y_test, results)
            plt.show()

    elif args.predict:
        if args.mmsi:
            df = predictor.load_and_prepare_data(args.data)
            df_processed, features = predictor.feature_engineering(df)
            df_mmsi = df_processed[df_processed['MMSI'] == args.mmsi]
            if df_mmsi.empty:
                print(f"AUCUN NAVIRE TROUVÉ AVEC MMSI {args.mmsi}")
                sys.exit(1)
            X = df_mmsi[features]
            X_scaled = predictor.scaler.fit_transform(X)
            predictor.train_models(X_scaled, df_mmsi['VesselCategory'])
            model = list(predictor.best_models.values())[0]
            pred = model.predict(X_scaled)
            print(f"PRÉDICTION POUR LE NAVIRE {args.mmsi}: {pred[0]}")

        elif args.features:
            feature_list = list(map(float, args.features.split(',')))
            X = np.array([feature_list])
            X_scaled = predictor.scaler.fit_transform(X)
            predictor.train_models(X_scaled, ['Cargo'])
            model = list(predictor.best_models.values())[0]
            pred = model.predict(X_scaled)
            print(f"PRÉDICTION À PARTIR DES FEATURES : {pred[0]}")
        else:
            print("VEUILLEZ SPÉCIFIER UN MMSI OU DES FEATURES POUR LA PRÉDICTION.")
            parser.print_help()
            sys.exit(1)

    else:
        print("AUCUNE ACTION SPÉCIFIÉE. VEUILLEZ UTILISER --train, --evaluate OU --predict.")
        parser.print_help()

if __name__ == "__main__":
    main()