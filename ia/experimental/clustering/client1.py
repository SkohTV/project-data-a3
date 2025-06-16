import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
import plotly.express as px
import joblib



FILE_PATH = "../../data/export_IA.csv" # chemin vers le fichier CSV contenant les données AIS
FEATURES = ['LAT', 'LON', 'SOG', 'COG', 'Heading'] # les features à utiliser pour le clustering
SAMPLE_SIZE = 40000 # taille de l'échantillon pour le clustering (taille max car s'il y a moins de données, on prend tout)
# mettre SAMPLE_SIZE a None pour faire sur toutes les données
SAMPLE_AGGLOMERATIVE = 5000 # taille de l'échantillon pour l'algorithme agglomerative (pour éviter les problèmes de mémoire) car agglomerative est tres gourmand en mémoire
# mettre SAMPLE_AGGLOMERATIVE a None pour faire sur toutes les données (fortement déconseillé car prend rapidement beacoup de RAM)
SAMPLE_DBSCAN = 10000 # taille de l'échantillon pour DBSCAN (pour éviter les problèmes de mémoire)
# mettre SAMPLE_DBSCAN a None pour faire sur toutes les données (fortement déconseillé)
PTS_FOR_HTML_VIEW = 400000 # nb de points affichés sur la carte interactive html
# mettre PTS_FOR_HTML_VIEW a None pour faire sur toutes les données

class InitData:
    """
    
    Classe qui permet de charger les données AIS à partir d'un fichier CSV,
    de les nettoyer et de les préparer pour le clustering.
    
    
    """
    def __init__(self, file_path, features):
        self.file_path = file_path
        self.features = features
        self.scaler = StandardScaler()
    
    def load_data(self):


        df = pd.read_csv(self.file_path)
        df_clean = df[self.features + ['VesselType'] ].dropna() # on garde dans tous les cas VesselType car c'est une information interressante si on veut comparer les résultats d'une prédiction avec la vérité
        # df_clean = df_clean[
        #     (df_clean['LAT'].between(-90, 90)) &
        #     (df_clean['LON'].between(-180, 180)) &
        #     (df_clean['SOG'] >= 0) & (df_clean['SOG'] <= 50)
        # ]
        # print(f"Donnees chargees: {len(df_clean):,} points")


        return df_clean
    
    def prepare_data(self, df, sample_size):
        

        # cette fonction permet de crop le fichier si on veut faire des entrainnements uniquements sur un échantillon
        # de plus il normalisera les data

        
        if sample_size != None and len(df) > sample_size:
            df_sample = df.head(sample_size)
        else:
            df_sample = df
        X = self.scaler.fit_transform(df_sample[self.features]) # normalisation des données
        return df_sample, X

class ClusterTester:
    """
    
    Classe qui permet de tester différents algorithmes de clustering (KMeans, Agglomerative, DBSCAN)
    et d'évaluer leurs performances en utilisant les métriques de silhouette, Calinski-Harabasz et Davies-Bouldin.
    Elle permet également de choisir le meilleur nombre de clusters pour KMeans en utilisant la méthode du coude.
    
    
    """
    def __init__(self):
        
        self.kmeans_inertias = [] # pour methode du coude 
        self.kmeans_silhouettes = []
        self.kmeans_calinski = []
        self.kmeans_davies = []
        self.k_range = list(range(2, 11)) # on testera pour un nombre de cluster de 2 à 10 pour chaque algo
        self.agglo_results = {}
        self.dbscan_results = {}
    
    def evaluate(self, X, labels):
        """
        Évalue les performances du clustering en utilisant les métriques de silhouette, Calinski-Harabasz et Davies-Bouldin.            

        Returns:
            tuple: (silhouette_score, calinski_harabasz_score, davies_bouldin_score)
        """


        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        return silhouette, calinski, davies
    
    def composite_score(self, sil, cal, dav):
        """
        Calcule un score composite basé sur les métriques de silhouette, Calinski-Harabasz et Davies-Bouldin.
        """


        # on normalise les 3 scores pour pouvoir y faire un score entre 0 et 1
        sil_norm = (sil + 1) / 2
        cal_norm = min(cal / 1000, 1)
        dav_norm = 1 / (1 + dav)
        return 0.4 * sil_norm + 0.3 * cal_norm + 0.3 * dav_norm
    
    def test_kmeans(self, X):
        
        """
        Teste l'algorithme KMeans sur les données X pour trouver le meilleur nombre de clusters (k) en utilisant la méthode du coude.

        Returns:
            tuple: (best_k, best_score) - le meilleur nombre de clusters et le score composite associé.
        """
        
        print("\nkmean:")
        best_k, best_score = 2, 0
        self.kmeans_inertias = []
        self.kmeans_silhouettes = []
        self.kmeans_calinski = []
        self.kmeans_davies = []
        
        for k in self.k_range:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            sil, cal, dav = self.evaluate(X, labels)
            score = self.composite_score(sil, cal, dav)
            
            self.kmeans_inertias.append(model.inertia_)
            self.kmeans_silhouettes.append(sil)
            self.kmeans_calinski.append(cal)
            self.kmeans_davies.append(dav)
            
            print(f"   k={k}: Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}, Composite={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"   Meilleur: k={best_k}")
        return best_k, best_score
    
    def test_agglomerative(self, X):
        """
        Teste l'algorithme AgglomerativeClustering sur les données X pour trouver le meilleur nombre de clusters (k) en utilisant les métriques de silhouette, Calinski-Harabasz et Davies-Bouldin.            

        Returns:
            tuple: (best_k, best_score) - le meilleur nombre de clusters et le score composite associé.
        """
        print("\nAgglomerative")
        
        # il faut réduir l'echantillon sinon on a des probleme de memoire
        if SAMPLE_AGGLOMERATIVE != None and len(X) > SAMPLE_AGGLOMERATIVE:
            X = X[:SAMPLE_AGGLOMERATIVE]
        
        best_k, best_score = 2, 0
        agglo_k_range = list(range(2, 8))
        agglo_sil, agglo_cal, agglo_dav, agglo_comp = [], [], [], []
        
        for k in agglo_k_range:
            
            
            model = AgglomerativeClustering(n_clusters=k) # création du modèle AgglomerativeClustering
            
            labels = model.fit_predict(X) # fit_predict pour obtenir les labels de clustering
            sil, cal, dav = self.evaluate(X, labels)
            score = self.composite_score(sil, cal, dav)
            
            agglo_sil.append(sil)
            agglo_cal.append(cal)
            agglo_dav.append(dav)
            agglo_comp.append(score)
            
            print(f"   k={k}: Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}, Composite={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        self.agglo_results = {
            'k_range': agglo_k_range,
            'silhouette': agglo_sil,
            'calinski': agglo_cal,
            'davies': agglo_dav,
            'composite': agglo_comp
        }
        
        print(f"   Meilleur: k={best_k}")
        return best_k, best_score
    
    def test_dbscan(self, X):
        """
        Teste l'algorithme DBSCAN sur les données X pour trouver les meilleurs paramètres (eps, min_samples) en utilisant les métriques de silhouette, Calinski-Harabasz et Davies-Bouldin.
        Returns:
            tuple: (best_params, best_score) - les meilleurs paramètres (eps, min_samples, n_clusters) et le score composite associé.
        """
        print("\ndbscan")

        if SAMPLE_DBSCAN != None and len(X) > SAMPLE_DBSCAN:
            X = X[:SAMPLE_DBSCAN]
            print(f"   Échantillon de {SAMPLE_DBSCAN} points")

        best_params, best_score = None, 0
        dbscan_configs = []
        
        for eps in [0.5, 0.7, 1.0]: # valeurs d'eps à tester (0.5, 0.7, 1.0 car les données sont normalisées)
            for min_samples in [10, 15]: # valeurs de min_samples à tester (10, 15 car les données sont normalisées)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    sil, cal, dav = self.evaluate(X[labels != -1], labels[labels != -1])
                    score = self.composite_score(sil, cal, dav)
                    
                    dbscan_configs.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'silhouette': sil,
                        'calinski': cal,
                        'davies': dav,
                        'composite': score
                    })
                    
                    print(f"   eps={eps}, min_samples={min_samples}: {n_clusters} clusters, Sil={sil:.3f}, Composite={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples, n_clusters)
        
        self.dbscan_results = dbscan_configs
        
        if best_params:
            print(f"   Meilleur: eps={best_params[0]}, min_samples={best_params[1]}")
        return best_params, best_score
    
    def plot_all_algorithms_analysis(self, kmeans_k, agglo_k, elbow_k):
        """
        Affiche les résultats des algorithmes de clustering EN PNG
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithmes de Clustering', fontsize=16, fontweight='bold')
        
        
        
        # coude pour kmean
        axes[0,0].plot(self.k_range, self.kmeans_inertias, 'bo-', linewidth=2, markersize=6)
        axes[0,0].axvline(x=elbow_k, color='r', linestyle='--', alpha=0.8, linewidth=2, label=f'Coude k={elbow_k}')
        axes[0,0].axvline(x=kmeans_k, color='g', linestyle='-', linewidth=2, label=f'Optimal k={kmeans_k}')
        axes[0,0].set_xlabel('Nombre de clusters (k)')
        axes[0,0].set_ylabel('Inertie')
        axes[0,0].set_title('KMeans - Methode du coude')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # les autres algo pour kmean
        sil_norm = np.array(self.kmeans_silhouettes)
        cal_norm = np.array(self.kmeans_calinski) / max(self.kmeans_calinski)
        dav_norm = 1 - (np.array(self.kmeans_davies) / max(self.kmeans_davies))
        
        axes[0,1].plot(self.k_range, sil_norm, 'go-', linewidth=2, markersize=6, label='Silhouette')
        axes[0,1].plot(self.k_range, cal_norm, 'ro-', linewidth=2, markersize=6, label='Calinski-Harabasz (norm.)')
        axes[0,1].plot(self.k_range, dav_norm, 'mo-', linewidth=2, markersize=6, label='Davies-Bouldin (inv.)')
        axes[0,1].axvline(x=kmeans_k, color='g', linestyle='-', linewidth=2, alpha=0.7, label=f'Optimal k={kmeans_k}')
        axes[0,1].set_xlabel('Nombre de clusters (k)')
        axes[0,1].set_ylabel('Scores normalisés (0-1)')
        axes[0,1].set_title('KMeans - Métriques normalisées')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        axes[0,1].set_ylim(0, 1.1)
        
        # composite pour kmean
        kmeans_composite = [self.composite_score(s, c, d) for s, c, d in 
                        zip(self.kmeans_silhouettes, self.kmeans_calinski, self.kmeans_davies)]
        axes[0,2].plot(self.k_range, kmeans_composite, 'ro-', linewidth=2, markersize=6)
        axes[0,2].axvline(x=kmeans_k, color='g', linestyle='-', linewidth=2, label=f'Optimal k={kmeans_k}')
        axes[0,2].set_xlabel('Nombre de clusters (k)')
        axes[0,2].set_ylabel('Score Composite')
        axes[0,2].set_title('KMeans - Score Composite')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()
        
        
        
        # # dbscan
        # if self.dbscan_results:
            

        #     #heatmap
        #     eps_values = sorted(list(set([r['eps'] for r in self.dbscan_results])))
        #     min_samples_values = sorted(list(set([r['min_samples'] for r in self.dbscan_results])))
            
        #     # matrice pour le dbscan
        #     score_matrix = np.zeros((len(min_samples_values), len(eps_values)))
        #     cluster_matrix = np.zeros((len(min_samples_values), len(eps_values)))
            
        #     for result in self.dbscan_results:
        #         i = min_samples_values.index(result['min_samples'])
        #         j = eps_values.index(result['eps'])
        #         score_matrix[i, j] = result['composite']
        #         cluster_matrix[i, j] = result['n_clusters']
            
            
        #     im = axes[1,0].imshow(score_matrix, cmap='viridis', aspect='auto')
            
        #     axes[1,0].set_xticks(range(len(eps_values)))
        #     axes[1,0].set_xticklabels([f'{eps:.1f}' for eps in eps_values])
        #     axes[1,0].set_yticks(range(len(min_samples_values)))
        #     axes[1,0].set_yticklabels([f'{ms}' for ms in min_samples_values])
            
        #     # valeur dans les cells
        #     for i in range(len(min_samples_values)):
        #         for j in range(len(eps_values)):
        #             score = score_matrix[i, j]
        #             n_clusters = int(cluster_matrix[i, j])
        #             if score > 0:
        #                 text_color = 'white' if score < 0.5 else 'black'
        #                 axes[1,0].text(j, i, f'{score:.2f}\n({n_clusters}c)', 
        #                             ha='center', va='center', color=text_color, fontsize=8)
            
        #     axes[1,0].set_xlabel('eps')
        #     axes[1,0].set_ylabel('min_samples')
        #     axes[1,0].set_title('DBSCAN - Score composite par paramètres\n(nombre de clusters)')
            
            
        #     plt.colorbar(im, ax=axes[1,0], shrink=0.8, label='Score Composite')
        # else:
        #     axes[1,0].text(0.5, 0.5, 'Aucun résultat DBSCAN valide', 
        #                 ha='center', va='center', transform=axes[1,0].transAxes)
        #     axes[1,0].set_title('DBSCAN - Aucun résultat')
        
        
        
        # metriques pour agglomerative
        agglo_sil_norm = np.array(self.agglo_results['silhouette'])
        agglo_cal_norm = np.array(self.agglo_results['calinski']) / max(self.agglo_results['calinski'])
        agglo_dav_norm = 1 - (np.array(self.agglo_results['davies']) / max(self.agglo_results['davies']))
        
        axes[1,1].plot(self.agglo_results['k_range'], agglo_sil_norm, 'go-', linewidth=2, markersize=6, label='Silhouette')
        axes[1,1].plot(self.agglo_results['k_range'], agglo_cal_norm, 'ro-', linewidth=2, markersize=6, label='Calinski-Harabasz (norm.)')
        axes[1,1].plot(self.agglo_results['k_range'], agglo_dav_norm, 'mo-', linewidth=2, markersize=6, label='Davies-Bouldin (inv.)')
        axes[1,1].axvline(x=agglo_k, color='g', linestyle='-', linewidth=2, alpha=0.7, label=f'Optimal k={agglo_k}')
        axes[1,1].set_xlabel('Nombre de clusters (k)')
        axes[1,1].set_ylabel('Scores normalisés (0-1)')
        axes[1,1].set_title('Agglomerative - Métriques normalisées')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        axes[1,1].set_ylim(0, 1.1)
        
        # composite pour agglomerative
        axes[1,2].plot(self.agglo_results['k_range'], self.agglo_results['composite'], 'ro-', linewidth=2, markersize=6)
        axes[1,2].axvline(x=agglo_k, color='g', linestyle='-', linewidth=2, label=f'Optimal k={agglo_k}')
        axes[1,2].set_xlabel('Nombre de clusters (k)')
        axes[1,2].set_ylabel('Score Composite')
        axes[1,2].set_title('Agglomerative - Score Composite')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('analyse.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        
        best_sil_idx = np.argmax(self.kmeans_silhouettes)
        best_cal_idx = np.argmax(self.kmeans_calinski)
        best_dav_idx = np.argmin(self.kmeans_davies)
        
        print("\npour kmean")
        print(f"Silhouette      : k={self.k_range[best_sil_idx]} (score: {self.kmeans_silhouettes[best_sil_idx]:.3f})")
        print(f"Calinski-Harabasz : k={self.k_range[best_cal_idx]} (score: {self.kmeans_calinski[best_cal_idx]:.1f})")
        print(f"Davies-Bouldin    : k={self.k_range[best_dav_idx]} (score: {self.kmeans_davies[best_dav_idx]:.3f})")
        print(f"Composite         : k={kmeans_k}")
        print(f"Coude            : k={elbow_k}")
        
        print("\npour agglomerartive")
        agglo_best_sil_idx = np.argmax(self.agglo_results['silhouette'])
        agglo_best_cal_idx = np.argmax(self.agglo_results['calinski'])
        agglo_best_dav_idx = np.argmin(self.agglo_results['davies'])
        
        print(f"Silhouette      : k={self.agglo_results['k_range'][agglo_best_sil_idx]} (score: {self.agglo_results['silhouette'][agglo_best_sil_idx]:.3f})")
        print(f"Calinski-Harabasz : k={self.agglo_results['k_range'][agglo_best_cal_idx]} (score: {self.agglo_results['calinski'][agglo_best_cal_idx]:.1f})")
        print(f"Davies-Bouldin    : k={self.agglo_results['k_range'][agglo_best_dav_idx]} (score: {self.agglo_results['davies'][agglo_best_dav_idx]:.3f})")
        print(f"Composite         : k={agglo_k}")



def clustering():
    
    """
    Fonction principale qui charge les données, prépare les échantillons, teste les algorithmes de clustering,
    affiche les résultats et permet à l'utilisateur de choisir un algorithme pour le clustering final
    """


    # on init le csv
    
    init = InitData(FILE_PATH, FEATURES)
    df = init.load_data()
    df_sample, X = init.prepare_data(df, SAMPLE_SIZE) # on crop et normalise
    
    # on test les algo
    tester = ClusterTester()
    kmeans_k, kmeans_score = tester.test_kmeans(X)
    agglo_k, agglo_score = tester.test_agglomerative(X)
    dbscan_params, dbscan_score = tester.test_dbscan(X)
    
    # on fait le coude pour l'affichage
    kl = KneeLocator(tester.k_range, tester.kmeans_inertias, curve="convex", direction="decreasing")
    elbow_k = kl.elbow
    
    # plot avant le choix de l'user (car c'est mieux de choisir quel algo prendre avec le analyse.png)
    tester.plot_all_algorithms_analysis(kmeans_k, agglo_k, elbow_k)
    

    print("\n\n\n")


    print(f"KMeans: k={kmeans_k}, Composite={kmeans_score:.3f}")
    print(f"Agglomerative: k={agglo_k}, Composite={agglo_score:.3f}")
    if dbscan_params:
        print(f"DBSCAN: k={dbscan_params[2]}, Composite={dbscan_score:.3f}")
    print(f"Coude: k={elbow_k} (methode du coude)")
    
    while True:
        choice = input("\nChoisir algorithme (kmeans/agglomerative/dbscan/coude): ").lower()
        if choice in ['kmeans', 'agglomerative', 'dbscan', 'coude']:
            break
        print("ca n'existe pas")
    
    X_full = init.scaler.transform(df[init.features])
    
    if choice == 'kmeans':
        model = KMeans(n_clusters=kmeans_k, random_state=42)
        clusters = model.fit_predict(X_full)
        
    elif choice == 'coude':
        model = KMeans(n_clusters=elbow_k, random_state=42)
        clusters = model.fit_predict(X_full)
        
    elif choice == 'agglomerative':
        # on utilise un echantillon car sinon il y a des probleme de RAM
        sample = df.head(min(30000, len(df)))
        X_sample = init.scaler.transform(sample[init.features])
        
        agglo = AgglomerativeClustering(n_clusters=agglo_k)
        sample_labels = agglo.fit_predict(X_sample)
        
        # on fait le centre avec Kmeans
        centers = np.array([X_sample[sample_labels == i].mean(axis=0) for i in range(agglo_k)])
        model = KMeans(n_clusters=agglo_k, init=centers, n_init=1)
        clusters = model.fit_predict(X_full)
        
    elif choice == 'dbscan':
        eps, min_samples, _ = dbscan_params
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(X_full)
    
    df['Cluster'] = clusters
    
    # affichage du pourcentatge de navire dans les clusters
    unique_clusters = sorted(np.unique(clusters))
    for cluster_id in unique_clusters:
        count = np.sum(clusters == cluster_id)
        perc = count / len(clusters) * 100
        print(f"   Cluster {cluster_id}: {count:,} navires ({perc:.1f}%)")
    
    # création de la carte interactive HTML
    size = PTS_FOR_HTML_VIEW if PTS_FOR_HTML_VIEW != None else len(df)

    df_map = df.head(min(size, len(df))).copy()
    df_map['Cluster'] = df_map['Cluster'].astype(str)
    
    fig = px.scatter_map(df_map, lat="LAT", lon="LON", color="Cluster",
                        hover_data=["SOG", "COG", "Heading", "VesselType"],
                        title=f"Clusters maritimes - {choice}")
    
    fig.write_html("carte_clusters.html")
    
    
    
    # on veux sauvegarder le model dans un seul fichier (on peut le faire en 2 fichiers mais c'est plus élégant en un seul, meme pour son utilisation dans le script je trouve que c'est mieux)
    
    pipeline = {
        'model': model,
        'scaler': init.scaler,
        'algorithm': choice,
        'features': FEATURES
    }
    joblib.dump(pipeline, 'model.pkl')
    
    df.to_csv('donnees_clusters.csv', index=False)
    

if __name__ == "__main__":
    clustering()
    


    print("pour tester le modele, il faut faire:")
    print("python3 script.py --lat 29.18 --lon -89.30 --sog 13.4 --cog 227.6 --heading 227")
    

