import streamlit as st
import numpy as np
import pandas as pd

# ==========================================
# 1. Core Prediction Engines (Data-Driven Coefficients)
# ==========================================

def predict_7_var_model(features):
    """
    Primary Model (7 Variables): Integrates continuous TILs.
    Exact Beta coefficients derived directly from SLN_Train_Set.csv
    Reference groups: Luminal_A, Grade G1, Location Center, LVI Negative.
    """
    intercept = -0.3285 
    
    # Order must strictly match the features array:
    # Age, Size(Continuous), LVI(Pos), Grade2, Grade3, Subtype(HER2, LumB, TNBC), Loc(ID, IU, OD, OU), TILs(Continuous)
    betas = np.array([
        -0.0129, # Age
         0.2938, # Tumor Size (Continuous, cm)
         0.6879, # LVI Positive
        -0.1273, # Grade G2
        -0.6308, # Grade G3
         0.3209, # Subtype HER2_Pos (Ref: Luminal_A)
         0.1875, # Subtype Luminal_B (Ref: Luminal_A)
        -0.1274, # Subtype TNBC (Ref: Luminal_A)
        -0.7557, # Loc InnerDown (Ref: Center)
        -0.8524, # Loc InnerUp (Ref: Center)
        -0.5242, # Loc OuterDown (Ref: Center)
        -0.2386, # Loc OuterUp (Ref: Center)
        -0.0108  # Stromal TILs (Continuous, %)
    ])
    
    z = intercept + np.dot(features, betas)
    return 1 / (1 + np.exp(-z))

def predict_6_var_model(features):
    """
    Baseline Model (6 Variables): Used when TILs status is Unclear/Missing.
    """
    intercept = -0.3939 
    
    # Order: Age, Size(Continuous), LVI(Pos), Grade2, Grade3, Subtype(HER2, LumB, TNBC), Loc(ID, IU, OD, OU)
    betas = np.array([
        -0.0126, # Age
         0.2930, # Tumor Size (Continuous, cm)
         0.6929, # LVI Positive
        -0.1379, # Grade G2
        -0.7268, # Grade G3
         0.2492, # Subtype HER2_Pos (Ref: Luminal_A)
         0.1574, # Subtype Luminal_B (Ref: Luminal_A)
        -0.2077, # Subtype TNBC (Ref: Luminal_A)
        -0.7722, # Loc InnerDown (Ref: Center)
        -0.8662, # Loc InnerUp (Ref: Center)
        -0.5203, # Loc OuterDown (Ref: Center)
        -0.2471  # Loc OuterUp (Ref: Center)
    ])
    
    z = intercept + np.dot(features, betas)
    return 1 / (1 + np.exp(-z))


# ==========================================
# 2. Page UI Configuration
# ==========================================
st.set_page_config(page_title="SLN Biological Risk Stratification", page_icon="🔬", layout="wide")
st.title("🔬 Breast Cancer SLN Metastasis: Biological Risk Stratification Platform")
st.caption("An investigational tool combining definitive postoperative clinicopathology and stromal TILs to identify biological profiles suitable for future de-escalation trials.")

# ==========================================
# 3. Sidebar Form (Clinical Input)
# ==========================================
with st.sidebar.form("medical_form"):
    st.header("📥 Definitive Pathology Data")
    
    age = st.number_input("1. Age (Years)", 18, 100, value=None, placeholder="Required")
    size = st.number_input("2. Tumor Size (cm)", 0.1, 15.0, value=None, placeholder="Required (e.g., 2.5)")
    
    lvi = st.selectbox("3. Lymphovascular Invasion (LVI)", ["Select", "Negative", "Positive"])
    grade = st.selectbox("4. Histological Grade", ["Select", "G1", "G2", "G3"])
    subtype = st.selectbox("5. Molecular Subtype", ["Select", "Luminal_A", "Luminal_B", "HER2_Pos", "TNBC"]) 
    loc = st.selectbox("6. Tumor Location", ["Select", "Center", "InnerUp", "InnerDown", "OuterUp", "OuterDown"])
    
    st.markdown("---")
    tils_status = st.selectbox("7. Stromal TILs Data", ["Available", "Unclear/Missing"])
    
    if tils_status == "Available":
        tils_val = st.number_input("↳ Stromal TILs (%)", 0.0, 100.0, value=None, placeholder="e.g., 15.0")
    else:
        tils_val = None
        
    submit_btn = st.form_submit_button("⚡ Calculate Biological Risk")

# ==========================================
# 4. Core Logic Processing & Output
# ==========================================
if submit_btn:
    # Mandatory field check
    if any(v is None for v in [age, size]) or "Select" in [lvi, grade, subtype, loc]:
        st.error("🚫 Error: All basic clinicopathological parameters (1-6) are required. Please complete the form.")
    elif tils_status == "Available" and tils_val is None:
        st.error("🚫 Error: Please enter the Stromal TILs percentage, or change status to 'Unclear/Missing'.")
    else:
        # Construct base features (Length: 12)
        base_features = [
            float(age), 
            float(size), 
            1 if lvi == "Positive" else 0, 
            1 if grade == "G2" else 0, 1 if grade == "G3" else 0,
            1 if subtype == "HER2_Pos" else 0, 1 if subtype == "Luminal_B" else 0, 1 if subtype == "TNBC" else 0,
            1 if loc == "InnerDown" else 0, 1 if loc == "InnerUp" else 0, 1 if loc == "OuterDown" else 0, 1 if loc == "OuterUp" else 0
        ]
        
        # Dynamic Model Switching based on TILs availability
        if tils_status == "Unclear/Missing":
            X = np.array(base_features)
            prob = predict_6_var_model(X)
            st.warning("⚠️ **Notice:** Because continuous TILs data is missing, this biological risk probability is computed using the baseline 6-variable clinical model.")
        else:
            X = np.array(base_features + [float(tils_val)]) # Length: 13
            prob = predict_7_var_model(X)

        # ==========================================
        # 5. Biological Risk Stratification & Research Implications
        # ==========================================
        if prob < 0.10:
            risk_tag = "Low Biological Risk"
            color = "#28a745" # Green
            advice = "💡 **Research & Stratification Insight:** Indicates a very low probability of SLN metastasis. In the context of contemporary de-escalation trials (e.g., SOUND, INSEMA), patients with this biological profile represent an ideal target population. This highlights the potential for future models, using strictly preoperative variables, to safely evaluate their candidacy for axillary omission."
        
        elif 0.10 <= prob <= 0.30:
            risk_tag = "Intermediate / Uncertain Risk"
            color = "#ff8c00" # Orange
            advice = "💡 **Research & Stratification Insight:** Indicates a transitional zone of risk. Decisions regarding axillary de-escalation for this biological profile remain uncertain and cannot be supported by this model alone. Comprehensive clinical judgment integrating patient age, strict imaging, tumor stage, molecular subtype, and patient preference is required."
        
        else:
            risk_tag = "High Biological Risk"
            color = "#dc3545" # Red
            advice = "💡 **Research & Stratification Insight:** Indicates a high probability of nodal burden. This biological profile strongly contradicts the rationale for axillary omission under current standard-of-care pathways. It reinforces the necessity of standard axillary staging (SLNB/ALND) for such patients."

        # ==========================================
        # 6. Result Rendering (HTML/CSS UI)
        # ==========================================
        st.markdown(f"""
            <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; color:white; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1 style="margin:0; font-size:4.5em;">{prob:.1%}</h1>
                <p style="font-size:1.5em; margin:0; font-weight:bold; letter-spacing: 1px;">Risk Stratification: {risk_tag}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(advice)

# ==========================================
# 7. Footer Disclaimer
# ==========================================
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This platform is designed as an **investigational biological risk stratification tool and research hypothesis generator**. It relies heavily on definitive postoperative pathology (including continuous TILs) to estimate nodal metastasis probability. Because these specific variables are fully acquired *after* the surgical event, this tool is **not** intended to direct real-time preoperative surgical decision-making (e.g., immediate SLNB omission). Instead, it serves to identify low-risk biological profiles to inform the development of future preoperative models.")
