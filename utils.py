import numpy as np
import pandas as pd

def parse_timestamp(timestamp_str):
    parts = str(timestamp_str).split(':')
    hour = int(parts[0])
    minute = int(parts[1])
    return hour, minute

def cyclical_encode(value, max_val):
    sin_val = np.sin(2 * np.pi * value / max_val)
    cos_val = np.cos(2 * np.pi * value / max_val)
    return sin_val, cos_val

def encode_hour_cyclical(hour):
    return cyclical_encode(hour, 24)

def encode_day_cyclical(day):
    return cyclical_encode(day, 7)

def calculate_r2_score(y_true, y_pred):

    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    competition_score = max(0, 100 * r2)
    return competition_score

def create_time_buckets(df, hour_column='hour', bucket_size=3):

    df = df.copy()
    df['time_bucket'] = df[hour_column] // bucket_size
    return df

def get_feature_correlation(df, target_column='demand', top_n=10):
  
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    correlations = correlations.drop(target_column)
    return correlations.head(top_n)

def print_model_summary(model, feature_names, model_name="Model"):

    print(f"\n{'='*50}")
    print(f"{model_name} Summary")
    print(f"{'='*50}")
    print(f"Model type: {type(model).__name__}")
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"Parameters: {params}")
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        print(f"\nTop 5 features:")
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

def clip_predictions(predictions, min_val=0, max_val=1):
    return np.clip(predictions, min_val, max_val)