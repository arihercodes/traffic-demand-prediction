import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import lightgbm as lgb
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Load Data 
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# Feature Engineering 
def add_basic_features(df):
    df = df.copy()
    # Geo prefixes for spatial hierarchy
    df['geo_prefix3'] = df['geohash'].str[:3]
    df['geo_prefix4'] = df['geohash'].str[:4]
    df['geo_prefix5'] = df['geohash'].str[:5]
    # Parse timestamp into hour/minute
    df['hour']         = df['timestamp'].str.split(':').str[0].astype(int)
    df['minute']       = df['timestamp'].str.split(':').str[1].astype(int)
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    # Cyclical time encoding (captures daily periodicity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # Peak hour flags
    df['is_rush_morning'] = ((df['hour'] >= 7)  & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    # Binary features
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin']     = (df['Landmarks'] == 'Yes').astype(int)
    return df

train = add_basic_features(train)
test  = add_basic_features(test)

# Label encode categoricals (fit on combined to handle all values)
cat_cols = ['RoadType', 'Weather', 'geo_prefix3', 'geo_prefix4', 'geo_prefix5', 'geohash']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].fillna('Unknown'), test[col].fillna('Unknown')])
    le.fit(combined)
    train[col + '_enc'] = le.transform(train[col].fillna('Unknown'))
    test[col  + '_enc'] = le.transform(test[col].fillna('Unknown'))

# Target Encoding / Aggregate Features 
# Geohash-level demand stats
geo_stats = train.groupby('geohash')['demand'].agg(['mean','std','max','min']).reset_index()
geo_stats.columns = ['geohash','geo_mean','geo_std','geo_max','geo_min']

# Same geohash × same timestamp from day 48 ("yesterday same time") — very predictive
geo_ts_d48 = train[train['day']==48].groupby(['geohash','timestamp'])['demand'].mean().reset_index()
geo_ts_d48.columns = ['geohash','timestamp','geo_ts_d48']

# Geohash × hour mean
geo_hour = train.groupby(['geohash','hour'])['demand'].mean().reset_index()
geo_hour.columns = ['geohash','hour','geo_hour_mean']

# Hour-of-day global mean
hour_agg = train.groupby('hour')['demand'].mean().reset_index()
hour_agg.columns = ['hour','hour_global_mean']

# geo_prefix4 mean (spatial neighborhood)
geo4_agg = train.groupby('geo_prefix4')['demand'].mean().reset_index()
geo4_agg.columns = ['geo_prefix4','geo4_mean']

def merge_aggs(df):
    df = df.merge(geo_stats,    on='geohash',              how='left')
    df = df.merge(geo_ts_d48,   on=['geohash','timestamp'],how='left')
    df = df.merge(geo_hour,     on=['geohash','hour'],      how='left')
    df = df.merge(hour_agg,     on='hour',                  how='left')
    df = df.merge(geo4_agg,     on='geo_prefix4',           how='left')
    return df

train = merge_aggs(train)
test  = merge_aggs(test)

# Prepare Matrices 
FEATURES = [
    'day','hour','minute','time_minutes','hour_sin','hour_cos',
    'is_rush_morning','is_rush_evening','is_night',
    'NumberofLanes','LargeVehicles_bin','Landmarks_bin','Temperature',
    'RoadType_enc','Weather_enc',
    'geo_prefix3_enc','geo_prefix4_enc','geo_prefix5_enc','geohash_enc',
    'geo_mean','geo_std','geo_max','geo_min',
    'geo_ts_d48','geo_hour_mean','hour_global_mean','geo4_mean',
]

X_train = train[FEATURES].copy()
y_train = train['demand'].copy()
X_test  = test[FEATURES].copy()

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test  = X_test.fillna(medians)

# LightGBM 
lgb_model = LGBMRegressor(
    objective='regression', metric='rmse',
    num_leaves=255, learning_rate=0.05,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_child_samples=20, reg_alpha=0.05, reg_lambda=0.1,
    n_estimators=2000, random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])

r2_lgb = r2_score(y_train, lgb_model.predict(X_train))
print(f"LightGBM — Train R²: {r2_lgb:.5f} | Score: {max(0,100*r2_lgb):.2f}")

# Random Forest 
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_leaf=5,
    max_features=0.7, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
r2_rf = r2_score(y_train, rf_model.predict(X_train))
print(f"Random Forest — Train R²: {r2_rf:.5f} | Score: {max(0,100*r2_rf):.2f}")

# Ensemble & Submit 
pred = np.clip(0.7 * lgb_model.predict(X_test) + 0.3 * rf_model.predict(X_test), 0, 1)

submission = pd.DataFrame({'Index': test['Index'], 'demand': pred})
submission.to_csv('submission.csv', index=False)
print(f"Submission saved — shape: {submission.shape}")
print(submission['demand'].describe())
