import streamlit as st
import numpy as np
import pandas as pd

# --- 1. Core Prediction Engines (Data-Driven Coefficients) ---

def predict_7_var_model(features):
    # Intercept derived from SLN_Train_Set.csv (7-variable model)
    intercept = -0.2235 
    
    # Betas derived from multivariable logistic regression (7 variables)
    betas = np.array([
        -0.0132, # 1. Age
         0.5431, # 2. Tumor Size > 2.0
         0.6826, # 3. LVI Positive
        -0.1417, # 4. Grade G2
        -0.6552, # 5. Grade G3
        -0.3562, # 6. Subtype Luminal_A
        -0.1408, # 7. Subtype Luminal_B
        -0.4711, # 8. Subtype TNBC
        -0.7125, # 9. Loc InnerDown
        -0.8260, # 10. Loc InnerUp
        -0.4841, # 11. Loc OuterDown
        -0.2002, # 12. Loc OuterUp
         0.3956, # 13. TILs Mod (vs High)
         0.6147  # 14. TILs Low (vs High)
    ])
    
    z = intercept + np.dot(features, betas)
    return 1 / (1 + np.exp(-z))

def predict_6_var_model(features):
    # Intercept derived from SLN_Train_Set.csv (6-variable baseline model, NO TILs)
    intercept = 0.2549 
    
    # Betas derived from multivariable logistic regression (6 variables)
    betas = np.array([
        -0.0127, # 1. Age
         0.5429, # 2. Tumor Size > 2.0
         0.6920, # 3. LVI Positive
        -0.1501, # 4. Grade G2
        -0.7530, # 5. Grade G3
        -0.2788, # 6. Subtype Luminal_A
        -0.1046, # 7. Subtype Luminal_B
        -0.4706, # 8. Subtype TNBC
        -0.7197, # 9. Loc InnerDown
        -0.8349, # 10. Loc InnerUp
        -0.4713, # 11. Loc OuterDown
        -0.2002  # 12. Loc OuterUp
    ])
    
    z = intercept + np.dot(features, betas)
    return 1 / (1 + np.exp(-z))

# --- 2. Page UI Configuration ---
st.set_page_config(page_title="SLN Decision Support System", layout="wide")
st.title("🏥 Breast Cancer SLN Metastasis Risk & Decision Support System")
st.caption("Based on machine learning predictive models, integrating SOUND/INSEMA trial de-escalation criteria.")

# --- 3. Sidebar Form ---
with st.sidebar.form("medical_form"):
    st.header("📥 Clinical & Pathological Data")
    
    age = st.number_input("1. Age (Years)", 18, 100, value=None, placeholder="Required")
    size = st.number_input("2. Tumor Size (cm)", 0.1, 5.0, value=None, placeholder="Required")
    
    lvi = st.selectbox("3. Lymphovascular Invasion (LVI)", ["Select", "Negative", "Positive"])
    grade = st.selectbox("4. Histological Grade", ["Select", "G1", "G2", "G3"])
    subtype = st.selectbox("5. Molecular Subtype", ["Select", "HER2_Pos", "Luminal_A", "Luminal_B", "TNBC"])
    loc = st.selectbox("6. Tumor Location", ["Select", "Center", "InnerUp", "InnerDown", "OuterUp", "OuterDown"])
    
    # Added "Unclear" option for missing data handling
    tils = st.selectbox("7. TILs Category", ["Select", "High", "Mod", "Low", "Unclear"])
    
    submit_btn = st.form_submit_button("⚡ Calculate Risk")

# --- 4. Logic Processing ---
if submit_btn:
    check_num = [age, size]
    check_select = [lvi, grade, subtype, loc, tils]
    
    if any(v is None for v in check_num) or "Select" in check_select:
        st.error("🚫 Error: All clinical parameters are required. Please complete the form.")
    else:
        # Base features (6 variables, present in both models)
        base_features = [
            age, 
            1 if size > 2.0 else 0,
            1 if lvi == "Positive" else 0, 
            1 if grade == "G2" else 0, 1 if grade == "G3" else 0,
            1 if subtype == "Luminal_A" else 0, 1 if subtype == "Luminal_B" else 0, 1 if subtype == "TNBC" else 0,
            1 if loc == "InnerDown" else 0, 1 if loc == "InnerUp" else 0, 1 if loc == "OuterDown" else 0, 1 if loc == "OuterUp" else 0
        ]
        
        # Determine which model to use dynamically
        if tils == "Unclear":
            # Use 6-variable model (Baseline)
            X = np.array(base_features)
            prob = predict_6_var_model(X)
            st.warning("⚠️ **Notice:** Because TILs status is 'Unclear', this risk probability is calculated using the baseline 6-variable clinical model.")
        else:
            # Use 7-variable model (Integrated with TILs)
            tils_features = [1 if tils == "Mod" else 0, 1 if tils == "Low" else 0]
            X = np.array(base_features + tils_features)
            prob = predict_7_var_model(X)

        # --- Risk Stratification & Recommendations ---
        if prob < 0.10:
            risk_tag = "Low Risk"
            color = "#28a745" # Green
            advice = "💡 **Recommendation: Consider omitting SLNB.** The predicted risk of metastasis is extremely low. Omitting axillary staging is unlikely to impact long-term survival."
        elif 0.10 <= prob <= 0.30:
            risk_tag = "Intermediate Risk"
            color = "#ff8c00" # Orange
            
            # SOUND/INSEMA trial logic integration
            is_eligible_exemption = (age >= 70 and size <= 2.0 and (subtype in ["Luminal_A", "Luminal_B"]))
            
            if is_eligible_exemption:
                advice = "💡 **Recommendation: Individualized Decision.** This patient meets the **SOUND/INSEMA trial** exemption criteria (≥70 years, T1, HR+/HER2-). SLNB omission may still be considered despite the intermediate risk."
            else:
                advice = "💡 **Recommendation: Standard SLNB.** The patient does not meet current trial criteria for omission. Standard surgical staging is recommended."
        else:
            risk_tag = "High Risk"
            color = "#dc3545" # Red
            advice = "💡 **Recommendation: Standard SLNB or ALND.** The predicted risk is high. Axillary staging is required."

        # --- Result Rendering ---
        st.markdown(f"""
            <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; color:white; margin-bottom: 20px;">
                <h1 style="margin:0; font-size:4em;">{prob:.1%}</h1>
                <p style="font-size:1.5em; margin:0; font-weight:bold;">Risk Stratification: {risk_tag}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(advice)

st.markdown("---")
st.caption("⚠️ This tool is based on a machine learning predictive model and is intended for research and clinical reference only. Final treatment decisions should be made by the attending physician considering the patient's specific circumstances and institutional guidelines.")