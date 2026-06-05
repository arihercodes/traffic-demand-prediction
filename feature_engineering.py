import pandas as pd
import numpy as np

def extract_geohash_features(df):
    df = df.copy()
    df['geo_prefix3'] = df['geohash'].str[:3]
    df['geo_prefix4'] = df['geohash'].str[:4]
    df['geo_prefix5'] = df['geohash'].str[:5]
    return df

def extract_temporal_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].str.split(':').str[0].astype(int)
    df['minute'] = df['timestamp'].str.split(':').str[1].astype(int)
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    return df

def add_cyclical_time_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def add_peak_hour_flags(df):
    df = df.copy()
    df['is_rush_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    return df

def add_binary_features(df):
    df = df.copy()
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
    return df

def create_aggregated_features(train, test):
 
    # Geohash-level statistics
    geo_stats = train.groupby('geohash')['demand'].agg(['mean', 'std', 'max', 'min']).reset_index()
    geo_stats.columns = ['geohash', 'geo_mean', 'geo_std', 'geo_max', 'geo_min']
    
    # Day 48 same timestamp (yesterday same time)
    geo_ts_d48 = train[train['day'] == 48].groupby(['geohash', 'timestamp'])['demand'].mean().reset_index()
    geo_ts_d48.columns = ['geohash', 'timestamp', 'geo_ts_d48']
    
    # Geohash × hour mean
    geo_hour = train.groupby(['geohash', 'hour'])['demand'].mean().reset_index()
    geo_hour.columns = ['geohash', 'hour', 'geo_hour_mean']
    
    # Hour-of-day global mean
    hour_agg = train.groupby('hour')['demand'].mean().reset_index()
    hour_agg.columns = ['hour', 'hour_global_mean']
    
    # Geo-prefix4 mean
    geo4_agg = train.groupby('geo_prefix4')['demand'].mean().reset_index()
    geo4_agg.columns = ['geo_prefix4', 'geo4_mean']
    
    return geo_stats, geo_ts_d48, geo_hour, hour_agg, geo4_agg

def merge_aggregated_features(df, geo_stats, geo_ts_d48, geo_hour, hour_agg, geo4_agg):
    df = df.copy()
    df = df.merge(geo_stats, on='geohash', how='left')
    df = df.merge(geo_ts_d48, on=['geohash', 'timestamp'], how='left')
    df = df.merge(geo_hour, on=['geohash', 'hour'], how='left')
    df = df.merge(hour_agg, on='hour', how='left')
    df = df.merge(geo4_agg, on='geo_prefix4', how='left')
    return df

def create_all_features(train, test):
    
    print("Extracting geohash features...")
    train = extract_geohash_features(train)
    test = extract_geohash_features(test)
    
    print("Extracting temporal features...")
    train = extract_temporal_features(train)
    test = extract_temporal_features(test)
    
    print("Adding cyclical time features...")
    train = add_cyclical_time_features(train)
    test = add_cyclical_time_features(test)
    
    print("Adding peak hour flags...")
    train = add_peak_hour_flags(train)
    test = add_peak_hour_flags(test)
    
    print("Adding binary features...")
    train = add_binary_features(train)
    test = add_binary_features(test)
    
    print("Creating aggregated features...")
    geo_stats, geo_ts_d48, geo_hour, hour_agg, geo4_agg = create_aggregated_features(train, test)
    
    print("Merging aggregated features...")
    train = merge_aggregated_features(train, geo_stats, geo_ts_d48, geo_hour, hour_agg, geo4_agg)
    test = merge_aggregated_features(test, geo_stats, geo_ts_d48, geo_hour, hour_agg, geo4_agg)
    
    return train, test