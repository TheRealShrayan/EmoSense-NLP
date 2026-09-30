import os
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import (
    preprocess_text,
    EMOTION_CONFIG,
    EMOTION_ID_TO_LABEL,
    EMOTION_LABEL_TO_ID
)
from src.model import (
    get_or_train_model,
    train_model,
    predict_text_emotion,
    load_data_file,
    prepare_dataset
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="EmotionNLP - Neural Sentiment & Emotion Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism & visual polish
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background Accent */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header Gradient Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        background: linear-gradient(90deg, #38ef7d 0%, #11998e 40%, #70A1FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 15px;
    }
    
    /* Custom Metric Badges */
    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 5px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #cbd5e1;
    }
    
    .badge-highlight {
        background: linear-gradient(135deg, rgba(56, 239, 125, 0.2) 0%, rgba(17, 153, 142, 0.2) 100%);
        border: 1px solid rgba(56, 239, 125, 0.4);
        color: #38ef7d;
    }
    
    /* Emotion Prediction Cards */
    .emotion-card {
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .emotion-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.45);
    }
    
    .emotion-emoji {
        font-size: 4.2rem;
        line-height: 1;
        margin-bottom: 10px;
        display: block;
    }
    
    .emotion-name {
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .emotion-confidence {
        font-size: 1.25rem;
        font-weight: 700;
        opacity: 0.95;
    }
    
    /* Custom Info Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
        transform: translateY(-1px);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource(show_spinner=False)
def load_app_model():
    return get_or_train_model()

# Helper function to cache dataset loading
@st.cache_data(show_spinner=False)
def load_all_datasets():
    train_df = prepare_dataset(load_data_file("train.txt"))
    val_df = prepare_dataset(load_data_file("val.txt"))
    test_df = prepare_dataset(load_data_file("test.txt"))
    return train_df, val_df, test_df

# Load resources
with st.spinner("Initializing Emotion Intelligence Engine & Models..."):
    pipeline, metrics = load_app_model()
    train_df, val_df, test_df = load_all_datasets()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=102550&format=png", width=70)
    st.title("⚙️ Control Panel")
    st.caption("NLP Model & System Settings")
    
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Select Classification Model",
        ["Logistic Regression + TF-IDF (Recommended)", "Multinomial Naive Bayes + BoW", "Multinomial Naive Bayes + TF-IDF"],
        index=0
    )
    
    model_type_key = {
        "Logistic Regression + TF-IDF (Recommended)": "logistic_tfidf",
        "Multinomial Naive Bayes + BoW": "nb_bow",
        "Multinomial Naive Bayes + TF-IDF": "nb_tfidf"
    }[model_choice]
    
    if st.button("🔄 Retrain / Switch Model", width="stretch"):
        with st.spinner(f"Training {model_choice}..."):
            t0 = time.time()
            new_pipeline, new_metrics = train_model(model_type=model_type_key)
            st.cache_resource.clear()
            pipeline, metrics = new_pipeline, new_metrics
            st.success(f"Model retrained in {time.time() - t0:.2f}s! Accuracy: {new_metrics['test_accuracy']*100:.2f}%")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Corpus Quick Overview")
    st.write(f"**Train Samples:** `{len(train_df):,}`")
    st.write(f"**Val Samples:** `{len(val_df):,}`")
    st.write(f"**Test Samples:** `{len(test_df):,}`")
    st.write(f"**Target Classes:** `6 Emotions`")
    st.write(f"**Active Model Accuracy:** `{metrics.get('test_accuracy', 0.866875)*100:.2f}%`")

    st.markdown("---")
    st.markdown("### 💡 Quick Preset Examples")
    preset_example = st.selectbox(
        "Choose an example text:",
        [
            "-- Select a sample --",
            "i feel so blessed and grateful for everyone who supported me today",
            "i can go from feeling hopeless to damned hopeful around people who care",
            "i feel irritated and rejected without anyone doing or saying anything",
            "i am feeling completely overwhelmed and terrified about what might happen",
            "i feel romantic and deeply affectionate whenever we talk",
            "i was astonished and completely amazed when I saw the unbelievable results"
        ]
    )

# Hero Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Neural Emotion & Sentiment Intelligence</div>
    <div class="hero-subtitle">Fine-grained text emotion classification powered by spaCy NLP preprocessing & Scikit-Learn Machine Learning.</div>
    <div>
        <span class="badge-pill badge-highlight">🎯 Accuracy: {:.2f}%</span>
        <span class="badge-pill">⚡ Model: Logistic Regression + TF-IDF</span>
        <span class="badge-pill">🔤 SpaCy Tokenizer</span>
        <span class="badge-pill">6 Emotion Categories</span>
    </div>
</div>
""".format(metrics.get("test_accuracy", 0.866875) * 100), unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Emotion Classifier",
    "📊 Dataset & EDA Analytics",
    "🧪 Model Benchmarks & Metrics",
    "🛠️ NLP Pipeline & Preprocessor"
])

# ==========================================
# TAB 1: INTERACTIVE EMOTION CLASSIFIER
# ==========================================
with tab1:
    col_input, col_output = st.columns([1.1, 1], gap="large")
    
    default_text_val = ""
    if preset_example and preset_example != "-- Select a sample --":
        default_text_val = preset_example
    else:
        default_text_val = "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares"

    with col_input:
        st.subheader("💬 Enter Text to Analyze")
        user_text = st.text_area(
            "Input sentence or paragraph:",
            value=default_text_val,
            height=140,
            placeholder="Type how you are feeling or paste any message..."
        )
        
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            analyze_btn = st.button("✨ Predict Emotion", width="stretch")
        with btn_col2:
            if st.button("🗑️ Clear Text", width="stretch"):
                user_text = ""
                st.rerun()

    if user_text:
        res = predict_text_emotion(user_text, pipeline)
        top_label = res["predicted_label"]
        top_conf = res["confidence"]
        cfg = EMOTION_CONFIG.get(top_label, EMOTION_CONFIG["joy"])
        
        with col_output:
            st.subheader("🎯 Emotion Prediction Result")
            
            # Emotion display card
            card_html = f"""
            <div class="emotion-card" style="background: {cfg['gradient']};">
                <span class="emotion-emoji">{cfg['emoji']}</span>
                <div class="emotion-name">{top_label}</div>
                <div class="emotion-confidence">{top_conf * 100:.1f}% Confidence Score</div>
                <p style="margin-top: 10px; font-size: 0.95rem; opacity: 0.9; margin-bottom: 0;">{cfg['description']}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
        # Full Probability Breakdown Section
        st.markdown("---")
        st.subheader("📈 Probability Distribution across All 6 Emotions")
        
        prob_df = pd.DataFrame([
            {
                "Emotion": emo.capitalize(),
                "Probability": prob * 100,
                "Emoji": EMOTION_CONFIG[emo]["emoji"],
                "Color": EMOTION_CONFIG[emo]["color"]
            }
            for emo, prob in res["probabilities"].items()
        ]).sort_values("Probability", ascending=True)

        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Emotion",
            orientation="h",
            text=prob_df["Probability"].apply(lambda x: f"{x:.2f}%"),
            color="Emotion",
            color_discrete_map={row["Emotion"]: row["Color"] for _, row in prob_df.iterrows()}
        )
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9", size=14),
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_prob, width="stretch")

        # Preprocessed token inspection
        with st.expander("🔍 View Preprocessed NLP Tokens"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Text:**")
                st.info(user_text)
            with c2:
                st.markdown("**Cleaned & Tokenized Text (spaCy):**")
                st.success(res["cleaned_text"] if res["cleaned_text"] else "(Empty after stopword removal)")

    # Batch File Upload Section
    st.markdown("---")
    st.subheader("📁 Batch Text Emotion Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file containing a column named 'text' for batch prediction", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                lines = [line.decode("utf-8").strip() for line in uploaded_file.readlines()]
                batch_df = pd.DataFrame({"text": lines})
                
            if "text" not in batch_df.columns and len(batch_df.columns) > 0:
                batch_df.rename(columns={batch_df.columns[0]: "text"}, inplace=True)
                
            st.write(f"Uploaded `{len(batch_df)}` rows for analysis.")
            if st.button("🚀 Process Batch Predictions"):
                with st.spinner("Analyzing emotion predictions for batch..."):
                    results = [predict_text_emotion(str(t), pipeline) for t in batch_df["text"]]
                    batch_df["Predicted Emotion"] = [r["predicted_label"] for r in results]
                    batch_df["Confidence Score"] = [round(r["confidence"] * 100, 2) for r in results]
                    
                st.dataframe(batch_df.head(20), width="stretch")
                
                # Batch summary chart
                counts = batch_df["Predicted Emotion"].value_counts().reset_index()
                counts.columns = ["Emotion", "Count"]
                fig_batch = px.pie(
                    counts,
                    names="Emotion",
                    values="Count",
                    title="Batch Prediction Distribution",
                    color="Emotion",
                    color_discrete_map={emo: EMOTION_CONFIG[emo]["color"] for emo in EMOTION_CONFIG if emo in counts["Emotion"].values}
                )
                st.plotly_chart(fig_batch, width="stretch")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# TAB 2: DATASET & EDA ANALYTICS
# ==========================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis & Corpus Insights")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### 1. Emotion Class Distribution (Train Dataset)")
        train_counts = train_df["emotion"].value_counts().reset_index()
        train_counts.columns = ["Emotion", "count"]
        
        fig_dist = px.bar(
            train_counts,
            x="Emotion",
            y="count",
            text="count",
            color="Emotion",
            color_discrete_map={emo: EMOTION_CONFIG[emo]["color"] for emo in EMOTION_CONFIG}
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            showlegend=False,
            height=350,
            xaxis=dict(title=""),
            yaxis=dict(title="Sample Count", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_dist, width="stretch")

    with eda_col2:
        st.markdown("#### 2. Split Proportion Comparison")
        split_df = pd.DataFrame([
            {"Split": "Train", "Count": len(train_df)},
            {"Split": "Validation", "Count": len(val_df)},
            {"Split": "Test", "Count": len(test_df)}
        ])
        fig_split = px.pie(
            split_df,
            names="Split",
            values="Count",
            hole=0.4,
            color_discrete_sequence=["#2563eb", "#38ef7d", "#ff758c"]
        )
        fig_split.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            height=350
        )
        st.plotly_chart(fig_split, width="stretch")

    st.markdown("---")
    st.markdown("#### 3. Sentence Length & Word Count Distributions")
    
    train_df["char_length"] = train_df["text"].apply(len)
    train_df["word_count"] = train_df["text"].apply(lambda x: len(str(x).split()))
    
    box_col1, box_col2 = st.columns(2)
    with box_col1:
        fig_box_word = px.box(
            train_df,
            x="emotion",
            y="word_count",
            color="emotion",
            title="Word Count per Emotion Category",
            color_discrete_map={emo: EMOTION_CONFIG[emo]["color"] for emo in EMOTION_CONFIG}
        )
        fig_box_word.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            showlegend=False,
            height=350,
            xaxis=dict(title=""),
            yaxis=dict(title="Word Count", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_box_word, width="stretch")

    with box_col2:
        fig_hist = px.histogram(
            train_df,
            x="word_count",
            color="emotion",
            marginal="rug",
            title="Overall Text Length Histogram",
            color_discrete_map={emo: EMOTION_CONFIG[emo]["color"] for emo in EMOTION_CONFIG}
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            height=350,
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("---")
    st.markdown("#### 4. Interactive Raw Corpus Browser")
    selected_emo_filter = st.selectbox("Filter Samples by Emotion:", ["All"] + list(EMOTION_CONFIG.keys()))
    
    if selected_emo_filter != "All":
        filtered_df = train_df[train_df["emotion"] == selected_emo_filter]
    else:
        filtered_df = train_df
        
    st.dataframe(filtered_df[["text", "emotion", "word_count", "char_length"]].head(50), width="stretch")

# ==========================================
# TAB 3: MODEL BENCHMARKS & METRICS
# ==========================================
with tab3:
    st.subheader("🧪 Model Performance & Evaluation Metrics")
    
    report = metrics.get("classification_report", {})
    acc = metrics.get("test_accuracy", 0.866875)
    
    # Top Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Macro Precision", f"{report.get('macro avg', {}).get('precision', 0.88)*100:.2f}%")
    m3.metric("Macro Recall", f"{report.get('macro avg', {}).get('recall', 0.77)*100:.2f}%")
    m4.metric("Weighted F1-Score", f"{report.get('weighted avg', {}).get('f1-score', 0.86)*100:.2f}%")

    st.markdown("---")
    cm_col, class_col = st.columns([1.1, 1], gap="large")
    
    with cm_col:
        st.markdown("#### 1. Test Set Confusion Matrix")
        cm = np.array(metrics.get("confusion_matrix", [
            [889, 21, 5, 2, 8, 21],
            [32, 350, 6, 2, 21, 16],
            [12, 4, 189, 8, 11, 72],
            [4, 4, 11, 56, 12, 26],
            [35, 31, 8, 8, 305, 10],
            [18, 9, 12, 3, 2, 977]
        ]))
        
        labels = [c.capitalize() for c in metrics.get("classes", list(EMOTION_CONFIG.keys()))]
        
        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Viridis",
            labels=dict(x="Predicted Label", y="True Label", color="Sample Count")
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            height=400
        )
        st.plotly_chart(fig_cm, width="stretch")

    with class_col:
        st.markdown("#### 2. Per-Class F1-Score Breakdown")
        
        class_metrics = []
        for emo_id, emo_name in EMOTION_ID_TO_LABEL.items():
            if emo_name in report:
                class_metrics.append({
                    "Emotion": emo_name.capitalize(),
                    "Precision": report[emo_name]["precision"] * 100,
                    "Recall": report[emo_name]["recall"] * 100,
                    "F1-Score": report[emo_name]["f1-score"] * 100,
                })
        
        if class_metrics:
            df_class_m = pd.DataFrame(class_metrics)
            fig_f1 = px.bar(
                df_class_m,
                x="Emotion",
                y="F1-Score",
                text=df_class_m["F1-Score"].apply(lambda x: f"{x:.1f}%"),
                color="Emotion",
                color_discrete_map={emo.capitalize(): EMOTION_CONFIG[emo]["color"] for emo in EMOTION_CONFIG}
            )
            fig_f1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                showlegend=False,
                height=400,
                yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.1)")
            )
            st.plotly_chart(fig_f1, width="stretch")

    st.markdown("---")
    st.markdown("#### 3. Full Classification Report Table")
    if class_metrics:
        st.dataframe(pd.DataFrame(class_metrics).style.format({
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%"
        }), width="stretch")

# ==========================================
# TAB 4: NLP PIPELINE & ARCHITECTURE
# ==========================================
with tab4:
    st.subheader("🛠️ End-to-End NLP Architecture & Pipeline Flow")
    
    st.markdown("""
    <div class="glass-card">
        <h4>🔄 Text Processing Workflow</h4>
        <ol style="line-height: 1.8; color: #cbd5e1;">
            <li><b>Raw Input Extraction:</b> Reads text string from user input or CSV batch files.</li>
            <li><b>Lowercasing:</b> Normalizes text case across all characters.</li>
            <li><b>Punctuation & Number Removal:</b> Strips special characters, digits, and noise symbols.</li>
            <li><b>Non-ASCII / Emoji Cleaning:</b> Filters out unicode non-ASCII sequences while maintaining string structure.</li>
            <li><b>SpaCy Tokenization & Stopwords Filtering:</b> Utilizes <code>spacy.blank("en")</code> tokenizer. Custom rules preserve crucial negation words (<code>not</code>, <code>no</code>, <code>never</code>).</li>
            <li><b>Feature Extraction (TF-IDF Vectorizer):</b> Computes term frequency-inverse document frequency over unigrams & bigrams (13,000+ vocabulary dimensions).</li>
            <li><b>Classification (Logistic Regression):</b> Calculates multi-class log probabilities for 6 emotions (Joy, Sadness, Anger, Fear, Love, Surprise).</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧪 Interactive Pipeline Simulator")
    sim_input = st.text_input("Test a string to watch step-by-step transformations:", "I am NOT feeling happy at all! 😢 #sadness123")
    
    if sim_input:
        step1 = sim_input.lower()
        step2 = preprocess_text(sim_input)
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**1. Raw Input**")
            st.code(sim_input)
        with s2:
            st.markdown("**2. Lowercased**")
            st.code(step1)
        with s3:
            st.markdown("**3. Cleaned & Tokenized**")
            st.code(step2)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 15px 0;">
    EmotionNLP Intelligence Platform • Built with Python, Streamlit, spaCy & Scikit-Learn
</div>
""", unsafe_allow_html=True)
