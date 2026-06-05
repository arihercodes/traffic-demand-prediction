import numpy as np

def ensemble_predictions(predictions, weights):
    predictions = np.array(predictions)
    weights = np.array(weights) / np.sum(weights)
    
    combined = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        combined += weights[i] * pred
    
    return combined

def create_ensemble(lgb_model, rf_model):

    class Ensemble:
        def __init__(self, lgb_model, rf_model, lgb_weight=0.7, rf_weight=0.3):
            self.lgb_model = lgb_model
            self.rf_model = rf_model
            self.lgb_weight = lgb_weight
            self.rf_weight = rf_weight
        
        def predict(self, X):
            lgb_pred = self.lgb_model.predict(X)
            rf_pred = self.rf_model.predict(X)
            
            # Weighted average
            combined = self.lgb_weight * lgb_pred + self.rf_weight * rf_pred
            
            return combined
        
        def predict_with_clip(self, X, min_val=0, max_val=1):
            pred = self.predict(X)
            return np.clip(pred, min_val, max_val)
    
    return Ensemble(lgb_model, rf_model, lgb_weight=0.7, rf_weight=0.3)

def get_ensemble_weights(r2_lgb, r2_rf):
    total = r2_lgb + r2_rf
    lgb_weight = r2_lgb / total
    rf_weight = r2_rf / total
    print(f"Calculated weights - LightGBM: {lgb_weight:.3f}, RF: {rf_weight:.3f}")
    return lgb_weight, rf_weight