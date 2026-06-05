import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import lightgbm as lgb
from lightgbm import LGBMRegressor

def train_lightgbm(X_train, y_train, params=None):
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 255,
            'learning_rate': 0.05,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1,
            'n_estimators': 2000,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    print("Training LightGBM model...")
    model = LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500)
        ]
    )
    
    train_pred = model.predict(X_train)
    r2 = r2_score(y_train, train_pred)
    print(f"LightGBM Train R²: {r2:.5f}")
    
    return model, r2

def train_random_forest(X_train, y_train, params=None):
    if params is None:
        params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_leaf': 5,
            'max_features': 0.7,
            'random_state': 42,
            'n_jobs': -1
        }
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    r2 = r2_score(y_train, train_pred)
    print(f"Random Forest Train R²: {r2:.5f}")
    
    return model, r2

def get_feature_importance(model, feature_names, model_name="Model"):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        
        print(f"\n{model_name} - Top 10 Features:")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        return dict(zip(feature_names, importance))
    else:
        print(f"{model_name} does not have feature_importances_ attribute")
        return None