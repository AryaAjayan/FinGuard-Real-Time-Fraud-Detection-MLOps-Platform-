import sys
import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure the parent directory is in the path so we can import train_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Streamlit Page Config
st.set_page_config(
    page_title="FinGuard – Real-Time Fraud Detection & MLOps Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS (Glassmorphism, Dark Mode, glowing colors)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #070a13;
        color: #f1f5f9;
    }
    
    /* Hero Title styling */
    .hero-container {
        background: radial-gradient(circle at top center, rgba(59, 130, 246, 0.18), rgba(16, 185, 129, 0.04));
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.6);
    }
    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3.6rem;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.4rem;
        max-width: 900px;
        margin: 0 auto;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }
    
    /* Highlighted Real-Time Card */
    .highlight-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 26px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        color: #38bdf8;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Glowing Badges */
    .badge-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
        animation: pulse 1.5s infinite;
    }
    .badge-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.6);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud_10k.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
API_URL = "http://127.0.0.1:8000"

# Header Hero Section
st.markdown("""
<div class='hero-container'>
    <div class='main-title'>FinGuard</div>
    <div class='sub-title'>Real-Time Fraud Detection & MLOps Platform</div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Controls
st.sidebar.markdown("### 🛠️ FinGuard System Controls")

# Check dataset status
has_data = os.path.exists(DATA_PATH)
if has_data:
    st.sidebar.success("📊 Dataset: credit_card_fraud_10k.csv found!")
else:
    st.sidebar.warning("⚠️ Dataset: credit_card_fraud_10k.csv is missing")
    if st.sidebar.button("⚡ Generate Synthetic Dataset"):
        from data.generate_dataset import generate_exact_dataset
        with st.spinner("Generating 10,000 synthetic transaction records..."):
            generate_exact_dataset()
            st.rerun()

# Model loading & training status
has_metrics = os.path.exists(METRICS_PATH)
metrics_data = None
if has_metrics:
    try:
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    except Exception:
        pass

# Separate the system into exactly 2 pages using Streamlit Tabs
tab_predict, tab_mlops = st.tabs(["🛡️ Real-Time Fraud Detection", "⚙️ MLOps & Training Center"])

# ==========================================
# PAGE 1: Real-Time Fraud Detection
# ==========================================
with tab_predict:
    st.markdown("<h3 style='color: #60a5fa;'>🔍 Transaction Risk Analyzer</h3>", unsafe_allow_html=True)
    st.markdown("Enter transaction features below to compute instant machine learning risk probabilities from the serving model.")
    
    # HIGHLIGHTED Fraud Detection Form
    st.markdown("""
    <div class='highlight-card'>
        <div class='section-header'>💳 Real-Time Inference Interface</div>
    """, unsafe_allow_html=True)
    
    with st.form("inference_form"):
        col_inf1, col_inf2 = st.columns(2)
        
        with col_inf1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=84.47, step=1.0)
            transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 22)
            merchant_category = st.selectbox("Merchant Category", ["Food", "Clothing", "Electronics", "Grocery", "Travel"])
            device_trust_score = st.slider("Device Trust Score (25-99)", 25, 99, 66)
            
        with col_inf2:
            foreign_transaction = st.selectbox("Foreign Transaction", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            location_mismatch = st.selectbox("Location Mismatch", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            velocity_last_24h = st.slider("Velocity Last 24 Hours (0-9)", 0, 9, 3)
            cardholder_age = st.slider("Cardholder Age (18-69)", 18, 69, 40)
            
        submit = st.form_submit_button("🛡️ ANALYZE RISK NOW")
        
        if submit:
            data = {
                "amount": amount,
                "transaction_hour": transaction_hour,
                "merchant_category": merchant_category,
                "foreign_transaction": foreign_transaction,
                "location_mismatch": location_mismatch,
                "device_trust_score": device_trust_score,
                "velocity_last_24h": velocity_last_24h,
                "cardholder_age": cardholder_age
            }
            
            with st.spinner("Running deep pattern analysis..."):
                try:
                    res = requests.post(f"{API_URL}/predict", json=data)
                    response_json = res.json()
                    
                    if "error" in response_json:
                        st.error(response_json["error"])
                    else:
                        prob = response_json["fraud_probability"]
                        risk = response_json["risk_level"]
                        
                        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
                        st.subheader("🛡️ FinGuard Risk Report")
                        
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("Inference Fraud Probability", f"{prob*100:.2f}%")
                        with col_res2:
                            if risk == "HIGH":
                                st.markdown("<div class='badge-high'>🚨 RISK LEVEL: HIGH FRAUD PATTERN DETECTED</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='badge-low'>✅ RISK LEVEL: LOW (Legitimate Transaction)</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Inference serving endpoint unreachable at {API_URL}. Details: {e}")
                    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 2: MLOps & Training Center
# ==========================================
with tab_mlops:
    st.markdown("<h3 style='color: #34d399;'>⚙️ MLOps Pipelines & Training Center</h3>", unsafe_allow_html=True)
    st.markdown("Perform model training runs, upload new training datasets, and inspect diagnostic performance curves from one unified panel.")
    
    col_l, col_r = st.columns([1, 1.1])
    
    with col_l:
        # NEW SECTION: Dataset Ingestion File Uploader
        st.markdown("""
        <div class='glass-card'>
            <div class='section-header'>📥 Custom Dataset Ingestion</div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a new credit card fraud CSV dataset", 
            type=["csv"], 
            help="Required Columns: amount, transaction_hour, merchant_category, foreign_transaction, location_mismatch, device_trust_score, velocity_last_24h, cardholder_age, is_fraud"
        )
        
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                required_cols = ["amount", "transaction_hour", "merchant_category", "foreign_transaction", "location_mismatch", "device_trust_score", "velocity_last_24h", "cardholder_age", "is_fraud"]
                missing_cols = [col for col in required_cols if col not in new_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Invalid Schema! Missing columns: {', '.join(missing_cols)}")
                else:
                    new_df.to_csv(DATA_PATH, index=False)
                    st.success(f"✅ Successfully ingested custom dataset with {len(new_df):,} records!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")
                
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card'>
            <div class='section-header'>🚀 Model Training & Evaluation</div>
        """, unsafe_allow_html=True)
        
        # Training Button inside MLOps Center
        if has_data:
            if st.button("🚀 Execute XGBoost Training Run"):
                from model.train import train_model
                with st.spinner("Training model with scaled pos weights and logging to MLflow..."):
                    try:
                        metrics_data = train_model()
                        try:
                            requests.post(f"{API_URL}/reload")
                        except Exception:
                            pass
                        st.success("XGBoost Classifier successfully trained and logged!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        else:
            st.info("💡 Please generate or upload a dataset to enable model training.")
            
        if metrics_data is None:
            st.info("💡 No model metrics cached yet. Run the training script to begin.")
        else:
            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy Score", f"{metrics_data['accuracy']*100:.2f}%")
            m2.metric("ROC-AUC Score", f"{metrics_data.get('auc_roc', 0.95)*100:.2f}%")
            
            m3, m4 = st.columns(2)
            m3.metric("Precision Score", f"{metrics_data['precision']*100:.2f}%")
            m4.metric("Recall Score", f"{metrics_data['recall']*100:.2f}%")
            
            st.markdown("---")
            cm = metrics_data.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
            st.markdown(f"""
            **Evaluation Confusion Matrix:**
            *   True Negatives (TN): `{cm['tn']:,}`
            *   True Positives (TP): `{cm['tp']:,}`
            *   False Negatives (FN): `{cm['fn']:,}`
            *   False Positives (FP): `{cm['fp']:,}`
            """)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature Importance section
        st.markdown("""
        <div class='glass-card'>
            <div class='section-header'>⚡ XGBoost Feature Importances</div>
        """, unsafe_allow_html=True)
        
        if metrics_data is not None:
            feat_imp_dict = metrics_data.get("feature_importances", {})
            if feat_imp_dict:
                df_imp = pd.DataFrame(list(feat_imp_dict.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    df_imp, x="Importance", y="Feature",
                    orientation='h',
                    color="Importance",
                    color_continuous_scale="Blues",
                    height=280
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#f1f5f9"},
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train the model to visualize top feature importances.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_r:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-header'>📂 Transaction Dataset Diagnostics</div>
        """, unsafe_allow_html=True)
        
        if not has_data:
            st.info("💡 Generate a credit card fraud dataset inside the FinGuard System Controls to inspect statistics.")
        else:
            df_view = pd.read_csv(DATA_PATH)
            total_records = len(df_view)
            fraud_records = len(df_view[df_view["is_fraud"] == 1])
            fraud_pct = (fraud_records / total_records) * 100
            
            st.markdown(f"**Active Dataset:** `credit_card_fraud_10k.csv`")
            st.markdown(f"**Total Transaction Records:** `{total_records:,}`")
            st.markdown(f"**Fraud Cases Detected:** `{fraud_records:,}` (**{fraud_pct:.2f}%**)")
            
            st.markdown("---")
            st.markdown("#### 🔍 First 5 Raw Dataset Rows")
            st.dataframe(df_view.head(5), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 📊 Merchant Category Distribution")
            cat_counts = df_view["merchant_category"].value_counts().reset_index()
            cat_counts.columns = ["Merchant Category", "Count"]
            fig_pie = px.pie(
                cat_counts, names="Merchant Category", values="Count",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                legend={'font': {'color': "#f1f5f9"}},
                margin=dict(l=10, r=10, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
