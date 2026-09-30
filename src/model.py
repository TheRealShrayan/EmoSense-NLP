import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import preprocess_text, EMOTION_CONFIG, EMOTION_ID_TO_LABEL, EMOTION_LABEL_TO_ID

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.joblib")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load_data_file(file_name: str) -> pd.DataFrame:
    """Loads text and emotion labels from a semi-colon delimited file."""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, sep=";", header=None, names=["text", "emotion"])
    return df

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw dataframe text column matching main.ipynb integer encoding."""
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["label_id"] = df["emotion"].map(EMOTION_LABEL_TO_ID)
    return df

def build_pipeline(model_type: str = "logistic_tfidf") -> Pipeline:
    """
    Creates an ML pipeline combining Vectorizer and Classifier matching main.ipynb.
    Types supported:
    - 'logistic_tfidf': TF-IDF Vectorizer + Logistic Regression (~86.7% accuracy)
    - 'nb_bow': Count Vectorizer + Multinomial Naive Bayes
    - 'nb_tfidf': TF-IDF Vectorizer + Multinomial Naive Bayes
    """
    if model_type == "logistic_tfidf":
        return Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
    elif model_type == "nb_bow":
        return Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB(alpha=1.0))
        ])
    elif model_type == "nb_tfidf":
        return Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", MultinomialNB(alpha=1.0))
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def train_model(model_type: str = "logistic_tfidf", save_path: str = DEFAULT_MODEL_PATH) -> tuple[Pipeline, dict]:
    """Trains model on integer emotion labels (matching main.ipynb) and saves pipeline artifact."""
    print("Loading datasets...")
    train_df = prepare_dataset(load_data_file("train.txt"))
    val_df = prepare_dataset(load_data_file("val.txt"))
    test_df = prepare_dataset(load_data_file("test.txt"))

    pipeline = build_pipeline(model_type)
    
    print(f"Training {model_type} pipeline on {len(train_df)} training samples...")
    pipeline.fit(train_df["cleaned_text"], train_df["label_id"])
    
    print("Evaluating model...")
    y_test_pred = pipeline.predict(test_df["cleaned_text"])
    test_acc = float(accuracy_score(test_df["label_id"], y_test_pred))
    
    y_val_pred = pipeline.predict(val_df["cleaned_text"])
    val_acc = float(accuracy_score(val_df["label_id"], y_val_pred))
    
    ordered_labels = [EMOTION_ID_TO_LABEL[i] for i in range(6)]
    
    metrics = {
        "model_type": model_type,
        "test_accuracy": test_acc,
        "val_accuracy": val_acc,
        "classification_report": classification_report(
            test_df["label_id"],
            y_test_pred,
            target_names=ordered_labels,
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(test_df["label_id"], y_test_pred, labels=list(range(6))).tolist(),
        "classes": ordered_labels
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, save_path)
    print(f"Model successfully trained and saved to {save_path}. Test Accuracy: {test_acc:.4f}")
    
    return pipeline, metrics

def get_or_train_model(save_path: str = DEFAULT_MODEL_PATH) -> tuple[Pipeline, dict]:
    """Loads existing trained model artifact or retrains if not found or if legacy string model."""
    if os.path.exists(save_path):
        try:
            artifact = joblib.load(save_path)
            if isinstance(artifact, dict) and "pipeline" in artifact:
                pipe = artifact["pipeline"]
                # Ensure pipeline classes are integer IDs matching main.ipynb
                if hasattr(pipe, "classes_") and len(pipe.classes_) > 0:
                    if isinstance(pipe.classes_[0], (int, np.integer)):
                        return pipe, artifact.get("metrics", {})
        except Exception as e:
            print(f"Error loading model artifact from {save_path}: {e}. Retraining...")
    
    # If not found or legacy string model, force clean retrain
    return train_model(model_type="logistic_tfidf", save_path=save_path)

def predict_text_emotion(text: str, pipeline: Pipeline) -> dict:
    """
    Preprocesses text and returns predicted emotion, confidence score,
    and probability breakdown matching main.ipynb integer mapping.
    """
    cleaned = preprocess_text(text)
    
    if not cleaned.strip():
        return {
            "predicted_id": 5,
            "predicted_label": "joy",
            "confidence": 0.0,
            "probabilities": {EMOTION_ID_TO_LABEL[i]: 1.0 / 6 for i in range(6)},
            "cleaned_text": cleaned
        }
    
    probs = pipeline.predict_proba([cleaned])[0]
    prob_dict = {}
    for idx, class_id in enumerate(pipeline.classes_):
        label_str = EMOTION_ID_TO_LABEL.get(int(class_id), str(class_id))
        prob_dict[label_str] = float(probs[idx])
        
    best_idx = int(np.argmax(probs))
    best_class_id = int(pipeline.classes_[best_idx])
    best_label = EMOTION_ID_TO_LABEL.get(best_class_id, "unknown")
    
    return {
        "predicted_id": best_class_id,
        "predicted_label": best_label,
        "confidence": float(probs[best_idx]),
        "probabilities": prob_dict,
        "cleaned_text": cleaned
    }
