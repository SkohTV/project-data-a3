"""
    Script de préprocessing des données AIS en output du nettoyage avec R -> Pour préparer au calcul des features
"""

import pandas as pd
from datetime import datetime
from collections import Counter
import shutil
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import requests
import zipfile
import os
from io import BytesIO
from shapely.ops import transform
from functools import partial
import pyproj

GULF_BBOX = (-98, -80, 18, 31)  # GOLFE DU MEXIQUE BBOX
SPEED_THRESHOLD = 0.5  # SEUIL ARRÊT EN NŒUDS
DRAFT_CHANGE_THRESHOLD = 0.5  # SEUIL POUR DRAFT CHANGE EN MÈTRES
OFFSHORE_DISTANCE_THRESHOLD = 15000  # DISTANCE OFFSHORE
COAST_DISTANCE_THRESHOLD = 1500  # DISTANCE CÔTE


def ensure_datetime(df, date_column='BaseDateTime') -> pd.DataFrame:
    """SÉCURISATION DES COLONNES DE DATE"""
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S', utc=True)
    return df


def map_vessel_type(vessel_code) -> str | None:
    """MAPPING DES TYPES DE NAVIRES QUE L'ON VEUT PRÉDIRE"""
    if pd.isna(vessel_code):
        return None
    elif 60 <= vessel_code <= 69:
        return "Passenger"
    elif 70 <= vessel_code <= 79:
        return "Cargo"
    elif 80 <= vessel_code <= 89:
        return "Tanker"
    else:
        return None


def get_mode_vectorized(series) -> float | np.nan:
    """ON CHERCHE LE MODE D'UNE SÉRIE"""
    if series.empty:
        return np.nan
    return series.mode().iloc[0] if not series.mode().empty else series.iloc[0]


def download_natural_earth_data() -> gpd.GeoDataFrame | None:
    """TÉLÉCHARGEMENT DES DONNÉES ONSHORE DE NATURAL EARTH"""

    url = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
    
    try:
        print(f"-> DL DEPUIS {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall("temp_ne_data")
        
        onshore = gpd.read_file("temp_ne_data/ne_50m_land.shp")
        shutil.rmtree("temp_ne_data", ignore_errors=True)
        
        print("-> DONNÉES ONSHORE TÉLÉCHARGÉES ET EXTRACTÉES AVEC SUCCÈS.")
        return onshore
        
    except Exception as e:
        print(f"-> ÉCHEC DU DL : {e}")
        return None


def compute_distance_to_coast(points_gdf, onshore_gdf) -> np.ndarray:
    """CALCUL DES DISTANCES ONSHORE"""

    points_proj = points_gdf.to_crs("EPSG:3857")
    onshore_proj = onshore_gdf.to_crs("EPSG:3857")
    
    onshore_union = onshore_proj.unary_union
    
    distances = points_proj.geometry.distance(onshore_union)
    
    return distances.values


def compute_draft_delta(group_data) -> pd.DataFrame:
    """CALCUL DES DELTA DE DRAFT POUR UN GROUPE DE NAVIRES"""
    data = group_data.copy().sort_values('BaseDateTime')
    
    data['period_start_draft'] = data['Draft'].iloc[0]
    data['period_end_draft'] = data['Draft'].iloc[-1]
    
    time_diff = data['BaseDateTime'].iloc[-1] - data['BaseDateTime'].iloc[0]
    data['period_duration'] = time_diff.total_seconds() / 3600
    
    data['prev_period_end_draft'] = data.groupby('MMSI')['period_end_draft'].shift(1)
    data['draft_delta'] = (data['period_start_draft'] - data['prev_period_end_draft']).abs()
    data['significant_draft_change'] = (data['draft_delta'].notna() & (data['draft_delta'] > DRAFT_CHANGE_THRESHOLD))
    
    return data


def prepare_geodataframe(df, onshore_data, bbox=GULF_BBOX) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """PRÉPARATION DU GEODATAFRAME POUR LE CALCUL DES FONCTIONNALITÉS (OPTIMISATION)"""
    df = ensure_datetime(df)
    
    df_filtered = df[
        (df['LON'] >= bbox[0]) & (df['LON'] <= bbox[1]) &
        (df['LAT'] >= bbox[2]) & (df['LAT'] <= bbox[3])
    ].copy()
    
    df_gdf = gpd.GeoDataFrame(
        df_filtered,
        geometry=gpd.points_from_xy(df_filtered['LON'], df_filtered['LAT']),
        crs="EPSG:4326"
    )
    
    onshore_filtered = onshore_data.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    
    df_gdf['is_stopped'] = df_gdf['SOG'] <= SPEED_THRESHOLD
    
    onshore_union = onshore_filtered.unary_union
    df_gdf['on_land'] = df_gdf.geometry.within(onshore_union)
    
    df_gdf['distance_to_coast_m'] = compute_distance_to_coast(df_gdf, onshore_filtered)
    
    return df_gdf, onshore_filtered


def compute_segmentation(df) -> pd.DataFrame:
    """SEGMENTATION À PARTIR DES DONNÉES AIS"""
    df = ensure_datetime(df)
    
    df_work = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG']].copy()
    df_work = df_work.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
    
    df_work['is_stopped'] = df_work['SOG'] <= SPEED_THRESHOLD
    
    df_work['prev_stopped'] = df_work.groupby('MMSI')['is_stopped'].shift(1)
    df_work['state_change'] = (df_work['is_stopped'] != df_work['prev_stopped']).fillna(True)
    df_work['segment_id'] = df_work.groupby('MMSI')['state_change'].cumsum()
    
    segments = df_work.groupby(['MMSI', 'segment_id', 'is_stopped']).agg({
        'BaseDateTime': ['min', 'max'],
        'SOG': get_mode_vectorized,
        'LAT': ['first', 'last'],
        'LON': ['first', 'last']
    }).reset_index()
    
    segments.columns = ['MMSI', 'segment_id', 'is_stopped', 'start_time', 'end_time', 'cruise_speed', 'start_lat', 'end_lat', 'start_lon', 'end_lon']
    
    segments['duration_min'] = (segments['end_time'] - segments['start_time']).dt.total_seconds() / 60
    
    return segments[['MMSI', 'segment_id', 'is_stopped', 'start_time', 'end_time', 'duration_min', 'cruise_speed', 'start_lat', 'start_lon', 'end_lat', 'end_lon']]


def detect_segment_docking_status(segments, df_with_features, onshore_data) -> pd.DataFrame:
    """DÉTECTION DES STATUTS DE DOCKING"""
    
    columns_to_exclude = ["id", "VesselName", "IMO", "CallSign", "TransceiverClass", "COG", "Heading", "Length", "Width", "Cargo"]
    available_columns = [col for col in columns_to_exclude if col in df_with_features.columns and col != 'MMSI']
    df_clean = df_with_features.drop(columns=available_columns, errors='ignore').copy()
    
    df_clean = df_clean[df_clean['Draft'].round(2) == df_clean['Draft']].copy()
    
    df_gdf, _ = prepare_geodataframe(df_clean, onshore_data)
    
    df_gdf['likely_docked_point'] = (df_gdf['is_stopped'] & (df_gdf['on_land'] | (df_gdf['distance_to_coast_m'] <= COAST_DISTANCE_THRESHOLD)))
    
    df_gdf = df_gdf.groupby('MMSI', group_keys=False).apply(compute_draft_delta).reset_index(drop=True)
    
    df_gdf['likely_offshore_docked_point'] = (df_gdf['is_stopped'] & (~df_gdf['on_land']) & (df_gdf['distance_to_coast_m'] > OFFSHORE_DISTANCE_THRESHOLD) & df_gdf['significant_draft_change'])
    
    merge_df = df_clean[['MMSI', 'BaseDateTime']].reset_index().rename(columns={'index': 'original_index'})
    df_segmented = df_gdf.merge(merge_df, on=['MMSI', 'BaseDateTime'], how='left')
    
    df_segmented = df_segmented.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
    df_segmented['prev_stopped'] = df_segmented.groupby('MMSI')['is_stopped'].shift(1)
    df_segmented['state_change'] = (df_segmented['is_stopped'] != df_segmented['prev_stopped']).fillna(True)
    df_segmented['segment_id'] = df_segmented.groupby('MMSI')['state_change'].cumsum()
    
    segment_docking = df_segmented.groupby(['MMSI', 'segment_id']).agg({
        'likely_docked_point': 'any',
        'likely_offshore_docked_point': 'any',
        'distance_to_coast_m': 'mean'
    }).reset_index()
    
    segment_docking = segment_docking.rename(columns={
        'likely_docked_point': 'likely_docked',
        'likely_offshore_docked_point': 'likely_offshore_docked'
    })
    
    return segment_docking


def compute_average_distance_to_coast_by_travel(df, onshore_data, bbox=GULF_BBOX) -> pd.DataFrame:
    """CALCUL DES STATISTIQUES DE DISTANCE ONSHORE PAR NAVIRE"""
    df_gdf, _ = prepare_geodataframe(df, onshore_data, bbox)
    
    voyage_stats = df_gdf.groupby('MMSI').agg({'distance_to_coast_m': ['mean', 'std', 'count']}).reset_index()
    
    voyage_stats.columns = ['MMSI', 'distance_moyenne_cote_m', 'distance_std_cote_m', 'nb_points_trajet']
    
    numeric_columns = ['distance_moyenne_cote_m', 'distance_std_cote_m', 'nb_points_trajet']
    voyage_stats[numeric_columns] = voyage_stats[numeric_columns].round(2)
    
    return voyage_stats


if __name__ == "__main__":
    print("-> DÉMARRAGE DU TRAITEMENT DES DONNÉES...")
    
    print("-> CHARGEMENT DES DONNÉES CSV...")
    df = pd.read_csv("/root/data_ais/final.csv", sep=',')
    
    valid_vessel_types = list(range(60, 90))
    df = df[df['VesselType'].isin(valid_vessel_types)].copy()
    print(f"->  {len(df)} LIGNES APRÈS FILTRAGE DES TYPES DE NAVIRES VALABLES.")
    
    print("-> TÉLÉCHARGEMENT DES DONNÉES ONSHORE...")
    onshore_data = download_natural_earth_data()
    
    if onshore_data is not None:
        print("-> CALCUL DE LA SEGMENTATION DES TRAJECTOIRES...")
        segments = compute_segmentation(df)
        print(f"-> {len(segments)} SEGMENTS CALCULÉS.")
        print("\n-> APERÇU DES SEGMENTS :")
        print(segments.head())
        
        print("-> DÉTECTION DES STATUTS DE DOCKING.. (RELATIVEMENT LONGUE)")
        segment_docking = detect_segment_docking_status(segments, df, onshore_data)
        print(f"-> STATUTS DE DOCKING CALCULÉS POUR {len(segment_docking)} SEGMENTS.")
        print("\n-> APERÇU DES STATUTS DE DOCKING :")
        print(segment_docking.head())
        
        print("-> CALCUL DES STATISTIQUES DE DISTANCES ONSHORE PAR NAVIRE...")
        distances = compute_average_distance_to_coast_by_travel(df, onshore_data)
        print(f"-> DISTANCES CALCULÉES POUR {len(distances)} NAVIRES.")
        print("\n-> APERÇU DES DISTANCES :")
        print(distances.head())
        
        print("-> FUSION DES DONNÉES")
        segments_enriched = segments.merge(segment_docking, on=['MMSI', 'segment_id'], how='left')
        segments_enriched = segments_enriched.merge(distances, on='MMSI', how='left')
        
        vessel_features = df.groupby('MMSI').agg({'VesselType': lambda x: x.mode().iloc[0] if not x.mode().empty else None}).reset_index()
        
        segments_final = segments_enriched.merge(vessel_features, on='MMSI', how='left')
        
        write_path_final = "/root/project-data-a3/ia/data/segments_final.csv"
        segments_final.to_csv(write_path_final, index=False)
        print(f"-> SAUVEGARDE AU PATH SUIVANT : {write_path_final}")
        
        print(f"\n-> STATISTIQUES FINALES:")
        print(f"    - TOTAL SEGMENTS: {len(segments_final):,}")
        print(f"    - SEGMENTS STOPPED: {len(segments_final[segments_final['is_stopped']]):,}")
        print(f"    - SEGMENTS PROBABLEMENT DOCKÉ: {len(segments_final[segments_final['likely_docked'] == True]):,}")
        print(f"    - SEGMENTS PROBABLEMENT DOCKÉS OFFSHORE: {len(segments_final[segments_final['likely_offshore_docked'] == True]):,}")
        print("-> TRAITEMENT TERMINÉ!")
    else:
        print("-> LES DONNÉES CÔTIÈRES N'ONT PAS RÉUSSI À ÊTRE TÉLÉCHARGÉES = TRAITEMENT ABANDONNÉ.")