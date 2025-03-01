import joblib

# Load the model
model = joblib.load('../app/random_forest_fraud_best_model.pkl')

# Check if the model has a feature_names attribute (some models store this)
if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)
else:
    print("Model does not have feature_names_in_ attribute.")