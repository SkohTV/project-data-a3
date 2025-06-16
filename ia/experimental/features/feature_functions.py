"""
Ce script découle du script `feature_computing.py` et contient des fonctions pour calculer diverses caractéristiques à partir des données de segments et de standards.
Le but est d'obtenir le DataFrame final avec toutes les caractéristiques nécessaires pour la prédiction du Vessel Type.
"""

from feature_computing import *  
import pandas as pd
import numpy as np
from datetime import datetime


def initialize_result_df(df, mmsi_column='MMSI') -> pd.DataFrame:
    """FONCTION UTILITAIRE POUR GÉNÉRALISER L'INITIALISATION D'UN DataFrame DE RÉSULTAT AVEC LA COLONNE MMSI."""
    result_df = pd.DataFrame()
    result_df['MMSI'] = df[mmsi_column].unique()
    return result_df


def group_by_mmsi(df, mmsi_column='MMSI') -> pd.core.groupby.generic.DataFrameGroupBy:
    """FONCTION UTILITAIRE POUR GÉNÉRALISER LE GROUPEMENT PAR MMSI."""
    return df.groupby(mmsi_column)


def compute_mean_std_dist_travel(segments_df) -> pd.DataFrame:
    """CALCULE LA DISTANCE MOYENNE ET L'ÉCART-TYPE DE LA DISTANCE PARCOURUE POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(segments_df)
    one_knot = 1.852001/60 
    
    grouped = group_by_mmsi(segments_df)
    
    for mmsi, segments in grouped:
        moving_segments = segments[segments['is_stopped'] == False]
        
        if len(moving_segments) > 1:
            distances = moving_segments["cruise_speed"] * moving_segments["duration_min"] * one_knot
            
            if len(distances) > 0:
                mean_distance = distances.mean()
                std_distance = distances.std()
                
                result_df.loc[result_df['MMSI'] == mmsi, 'mean_distance_travel'] = mean_distance
                result_df.loc[result_df['MMSI'] == mmsi, 'std_distance_travel'] = std_distance

    return result_df[['MMSI', 'mean_distance_travel', 'std_distance_travel']]
    zero_mean_count = result_df['mean_distance_travel'].isna().sum()


def compute_mean_std_duration_stop(segments_df) -> pd.DataFrame: 
    """CALCULE LA DURÉE MOYENNE ET L'ÉCART-TYPE DES ARRÊTS POUR CHAQUE NAVIRE."""
    result_df = initialize_result_df(segments_df)
    grouped = group_by_mmsi(segments_df)
    
    for mmsi, segments in grouped:
        stopped_segments = segments[segments['is_stopped'] == True]
        
        if len(stopped_segments) > 1:
            start_times = pd.to_datetime(stopped_segments['start_time'])
            end_times = pd.to_datetime(stopped_segments['end_time'])
            
            stop_durations = (end_times - start_times).dt.total_seconds() / 60
            
            if len(stop_durations) > 0:
                mean_duration = stop_durations.mean()
                std_duration = stop_durations.std()
                
                result_df.loc[result_df['MMSI'] == mmsi, 'mean_duration_stop'] = mean_duration
                result_df.loc[result_df['MMSI'] == mmsi, 'std_duration_stop'] = std_duration
            
    return result_df[['MMSI', 'mean_duration_stop', 'std_duration_stop']]


def compute_mean_std_cruise_speed(df) -> pd.DataFrame:
    """CALCULE LA VITESSE MOYENNE ET L'ÉCART-TYPE DE LA VITESSE DE CROISIÈRE POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(df)
    grouped = group_by_mmsi(df)
    
    for mmsi, segments in grouped:
        cruise_speeds = segments[segments['is_stopped'] == False]['cruise_speed']
        
        if len(cruise_speeds) > 1:
            mean_speed = cruise_speeds.mean()
            std_speed = cruise_speeds.std()
            
            result_df.loc[result_df['MMSI'] == mmsi, 'mean_cruise_speed'] = mean_speed
            result_df.loc[result_df['MMSI'] == mmsi, 'std_cruise_speed'] = std_speed
    
    return result_df[['MMSI', 'mean_cruise_speed', 'std_cruise_speed']]


def compute_mean_std_dist_coast_travel(segments_df) -> pd.DataFrame:
    """CALCULE LA DISTANCE MOYENNE ET L'ÉCART-TYPE DE LA DISTANCE ONSHORE POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(segments_df)
    grouped = group_by_mmsi(segments_df)
    
    for mmsi, segments in grouped:
        moving_segments = segments[segments['is_stopped'] == False]
        
        if len(moving_segments) > 0:
            total_n = moving_segments['nb_points_trajet'].sum()
            
            if total_n > 0:
                mean_global = (moving_segments['distance_moyenne_cote_m'] * moving_segments['nb_points_trajet']).sum() / total_n
                std_global = np.sqrt((moving_segments['nb_points_trajet'] * (moving_segments['distance_std_cote_m']**2 + (moving_segments['distance_moyenne_cote_m'] - mean_global)**2)).sum() / total_n)
                
                result_df.loc[result_df['MMSI'] == mmsi, 'mean_dist_coast_travel'] = mean_global
                result_df.loc[result_df['MMSI'] == mmsi, 'std_dist_coast_travel'] = std_global
        
    return result_df[['MMSI', 'mean_dist_coast_travel', 'std_dist_coast_travel']]


def compute_mean_std_draft(standard_df) -> pd.DataFrame:
    """CALCULE LA PROFONDEUR MOYENNE ET L'ÉCART-TYPE DE LA PROFONDEUR POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(standard_df)
    grouped = group_by_mmsi(standard_df)
    
    for mmsi, df in grouped:
        if not df['Draft'].empty:
            mean_draft = df['Draft'].mean()
            std_draft = df['Draft'].std()
            
            result_df.loc[result_df['MMSI'] == mmsi, 'mean_draft'] = mean_draft
            result_df.loc[result_df['MMSI'] == mmsi, 'std_draft'] = std_draft
    
    return result_df[['MMSI', 'mean_draft', 'std_draft']]


def compute_number_occurence_travel(df) -> pd.DataFrame:
    """CALCULE LE NOMBRE D'OCCURRENCES DE TRAJETS POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(df)
    grouped = group_by_mmsi(df)
    
    for mmsi, segments in grouped:
        moving_segments = segments[segments['is_stopped'] == False]
        
        if not moving_segments['segment_id'].empty:
            number_occurence_travel = moving_segments['segment_id'].max() + 1
            result_df.loc[result_df['MMSI'] == mmsi, 'number_occurence_travel'] = number_occurence_travel
    
    return result_df[['MMSI', 'number_occurence_travel']]


def compute_number_occurence_docked(segments_df) -> pd.DataFrame:
    """CALCULE LE NOMBRE D'OCCURRENCES OÙ LE NAVIRE SEMBLE ÊTRE DOCKED."""
    
    result_df = initialize_result_df(segments_df)
    grouped = group_by_mmsi(segments_df)
    
    for mmsi, segments in grouped:
        stopped_segments = segments[segments['is_stopped'] == True]
        
        if not stopped_segments['likely_docked'].empty:
            number_occurence_docked = stopped_segments['likely_docked'].sum()
            result_df.loc[result_df['MMSI'] == mmsi, 'number_occurence_docked'] = number_occurence_docked
        
    return result_df[['MMSI', 'number_occurence_docked']]


def compute_number_occurence_off_shore_docked(segments_df) -> pd.DataFrame:
    """CALCULE LE NOMBRE D'OCCURRENCES OÙ LE NAVIRE SEMBLE ÊTRE OFFSHORE DOCKED."""
    
    result_df = initialize_result_df(segments_df)
    grouped = group_by_mmsi(segments_df)
    
    for mmsi, segments in grouped:
        stopped_segments = segments[segments['is_stopped'] == True]
        
        if not stopped_segments['likely_offshore_docked'].empty:
            number_occurence_off_shore_docked = stopped_segments['likely_offshore_docked'].sum()
            result_df.loc[result_df['MMSI'] == mmsi, 'number_occurence_off_shore_docked'] = number_occurence_off_shore_docked
        
    return result_df[['MMSI', 'number_occurence_off_shore_docked']]


def compute_onshore_duration_ratio(df) -> pd.DataFrame:
    """CALCULE LE RATIO DE DURÉE ONSHORE PAR RAPPORT À LA DURÉE TOTALE POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(df)
    grouped = group_by_mmsi(df)
    
    for mmsi, segments in grouped:
        total_duration = segments['duration_min'].sum()
        onshore_duration = segments[segments['is_stopped'] == True]['duration_min'].sum()
        
        if total_duration > 0:
            onshore_duration_ratio = onshore_duration / total_duration
            result_df.loc[result_df['MMSI'] == mmsi, 'onshore_duration_ratio'] = onshore_duration_ratio

    return result_df[['MMSI', 'onshore_duration_ratio']]


def compute_length_width_ratio(standard_df) -> pd.DataFrame:
    """CALCULE LE RATIO LONGUEUR/LARGEUR POUR CHAQUE NAVIRE."""
    
    result_df = initialize_result_df(standard_df)
    grouped = group_by_mmsi(standard_df)
    
    for mmsi, df in grouped:
        if not df['Length'].empty and not df['Width'].empty:
            length_width_ratio = df['Length'].mean() / df['Width'].mean()
            result_df.loc[result_df['MMSI'] == mmsi, 'length_width_ratio'] = length_width_ratio
    
    return result_df[['MMSI', 'length_width_ratio']]


def compute_directional_consistency(standard_df) -> pd.DataFrame:
    
    def angular_diff(hdg, cog):
        diff = np.abs(hdg - cog) % 360
        return np.where(diff > 180, 360 - diff, diff)
    
    def circular_consistency(series_deg):
        series_rad = np.deg2rad(series_deg.dropna())
        if len(series_rad) == 0:
            return np.nan
        R = np.sqrt(np.sum(np.cos(series_rad))**2 + np.sum(np.sin(series_rad))**2) / len(series_rad)
        return R

    result = initialize_result_df(standard_df)
    grouped = group_by_mmsi(standard_df)

    for mmsi, df in grouped:
        cog_data = df['COG'].dropna()
        heading_data = df['Heading'].dropna()
        
        if len(cog_data) > 0:
            COG_std = cog_data.std()
            result.loc[result['MMSI'] == mmsi, 'COG_std'] = COG_std
        else:
            result.loc[result['MMSI'] == mmsi, 'COG_std'] = np.nan
        
        if len(heading_data) > 0 and len(cog_data) > 0:
            min_len = min(len(heading_data), len(cog_data))
            hdg_cog_diff = angular_diff(heading_data.iloc[:min_len].values, cog_data.iloc[:min_len].values)
            result.loc[result['MMSI'] == mmsi, 'HDG_COG_diff_mean'] = np.mean(hdg_cog_diff)
            result.loc[result['MMSI'] == mmsi, 'HDG_COG_diff_std'] = np.std(hdg_cog_diff)
        else:
            result.loc[result['MMSI'] == mmsi, 'HDG_COG_diff_mean'] = np.nan
            result.loc[result['MMSI'] == mmsi, 'HDG_COG_diff_std'] = np.nan

        if len(heading_data) > 0:
            heading_consistency = circular_consistency(heading_data)
            result.loc[result['MMSI'] == mmsi, 'Heading_consistency'] = heading_consistency
        else:
            result.loc[result['MMSI'] == mmsi, 'Heading_consistency'] = np.nan
            
        if len(cog_data) > 0:
            cog_consistency = circular_consistency(cog_data)
            result.loc[result['MMSI'] == mmsi, 'COG_consistency'] = cog_consistency
        else:
            result.loc[result['MMSI'] == mmsi, 'COG_consistency'] = np.nan

    return result[['MMSI', 'COG_std', 'HDG_COG_diff_mean', 'HDG_COG_diff_std', 'Heading_consistency', 'COG_consistency']]


def compute_route_entropy(segments_df) -> pd.DataFrame:
    """PAS ENCORE IMPLÉMENTÉ."""
    return None


def compute_mean_turning_angle_variation_travel(df) -> pd.DataFrame:
    """PAS ENCORE IMPLÉMENTÉ."""
    return None

if __name__ == "__main__":
    
    segments_df = pd.read_csv('/root/project-data-a3/ia/data/segments_final.csv')
    standard_df = pd.read_csv('/root/data_ais/final.csv')
    
    result_std = compute_mean_std_draft(standard_df)
    result_seg = compute_mean_std_dist_travel(segments_df)
    result_speed = compute_mean_std_cruise_speed(segments_df)
    result_dist_coast = compute_mean_std_dist_coast_travel(segments_df)
    result_duration_stop = compute_mean_std_duration_stop(segments_df)
    result_nomber_occurence_travel = compute_number_occurence_travel(segments_df)
    result_length_width_ratio = compute_length_width_ratio(standard_df)
    result_directional_consistency = compute_directional_consistency(standard_df)
    result_number_occurence_docked = compute_number_occurence_docked(segments_df)
    result_number_occurence_off_shore_docked = compute_number_occurence_off_shore_docked(segments_df)
    result_onshore_duration_ratio = compute_onshore_duration_ratio(segments_df)
    
    dataframes = [
        result_std,
        result_seg,
        result_speed,
        result_dist_coast,
        result_duration_stop,
        result_nomber_occurence_travel,
        result_length_width_ratio,
        result_directional_consistency,
        result_number_occurence_docked,
        result_number_occurence_off_shore_docked,
        result_onshore_duration_ratio
    ]
    
    final_df = dataframes[0]
    for df in dataframes[1:]:
        final_df = pd.merge(final_df, df, on='MMSI', how='outer')
    
    print(f"NOMBRE TOTAL DE NAVIRES AVANT NETTOYAGE: {len(final_df)}")
    print(f"NOMBRE DE COLONNES AVANT NETTOYAGE: {len(final_df.columns)}")
    
    final_df_complete = final_df.dropna()
    
    print(f"NOMBRE TOTAL DE NAVIRES APRÈS NETTOYAGE: {len(final_df_complete)}")
    
    # Affichage du DataFrame final
    print("\nDATAFRAME FINAL:")
    print(final_df_complete.head())
    
    final_df_complete.to_csv('/root/project-data-a3/ia/data/features_completes.csv', index=False)