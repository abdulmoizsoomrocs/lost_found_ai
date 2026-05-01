import joblib

def save_model(model, vectorizer):
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("✅ Model and vectorizer saved!")