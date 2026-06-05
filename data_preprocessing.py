import pandas as pd
import numpy as np

def load_data(train_path='train.csv', test_path='test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test

def handle_missing_values(df, train_df=None):
    df = df.copy()
    
    # For test data, use training medians if provided
    if train_df is not None:
        medians = train_df.median()
    else:
        medians = df.median()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(medians[numeric_cols])
    
    # Fill categorical columns with mode or 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if train_df is not None:
            mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else 'Unknown'
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)
    
    return df

def remove_outliers(df, column='demand', method='iqr'):
    df = df.copy()
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"Removed {len(df) - len(filtered_df)} outliers from {column}")
        return filtered_df
    return df

def get_data_info(df, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"{name} Info")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    return None