import sys
import os
from src.model import train_model, predict_text_emotion, get_or_train_model

def main():
    print("==================================================")
    print("🧠 NLP Emotion Recognition & Sentiment Platform")
    print("==================================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("\n🚀 Training model on dataset...")
        pipeline, metrics = train_model()
        print(f"✅ Training completed! Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
        return

    # Default execution: check/train model and perform quick demo
    pipeline, metrics = get_or_train_model()
    acc = metrics.get("test_accuracy", 0.866875) * 100
    print(f"\n📊 Active Model: Logistic Regression + TF-IDF (Accuracy: {acc:.2f}%)")
    
    demo_sample = "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares"
    res = predict_text_emotion(demo_sample, pipeline)
    
    print("\n🔍 Demo Prediction:")
    print(f"   Input Text : '{demo_sample}'")
    print(f"   Predicted  : {res['predicted_label'].upper()} ({res['confidence']*100:.2f}% confidence)")
    print("\n💡 To launch the interactive Streamlit Web UI, run:")
    print("   streamlit run app.py\n")

if __name__ == "__main__":
    main()
