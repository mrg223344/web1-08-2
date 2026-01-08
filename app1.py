import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =====================================================
# 页面配置
# =====================================================
st.set_page_config(
    page_title="12-Hour Septic Shock Early Warning",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# 标题
# =====================================================
st.markdown("""
# 🩺 A 12-Hour Early Warning Model for Septic Shock in the ICU
**Random Forest–Based Clinical Decision Support Tool with SHAP Explanation**
""")

st.markdown("---")

# =====================================================
# 加载模型
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# =====================================================
# 输入界面（15 个模型特征）
# =====================================================
st.header("🧪 Key Clinical Variables (Within 12 Hours)")

col1, col2, col3 = st.columns(3)

with col1:
    Pneumonia = st.checkbox("Pneumonia")
    COPD = st.checkbox("COPD")
    age = st.number_input("Age (years)", 18, 120, 65)
    heartrate = st.number_input("Heart Rate (bpm)", 30, 200, 90)
    SBP = st.number_input("Systolic BP (mmHg)", 50, 250, 120)

with col2:
    respiratoryrate = st.number_input("Respiratory Rate (/min)", 5, 60, 20)
    spo2 = st.number_input("SpO₂ (%)", 50, 100, 96)
    temperature = st.number_input("Temperature (°C)", 33.0, 42.0, 36.8)
    WBC = st.number_input("WBC (×10⁹/L)", 0.1, 100.0, 8.0)
    Albumin = st.number_input("Albumin (g/dL)", 1.0, 6.0, 3.5)

with col3:
    ALT = st.number_input("ALT (U/L)", 0, 10000, 30)
    BUN = st.number_input("BUN (mg/dL)", 1, 300, 20)
    sodium = st.number_input("Sodium (mmol/L)", 110, 160, 135)
    Plateletcount = st.number_input("Platelet Count (×10⁹/L)", 5, 1000, 200)
    SOFA = st.number_input("SOFA Score", 0, 24, 6)

st.markdown("---")

# =====================================================
# 构造输入 DataFrame（严格对齐模型特征）
# =====================================================
input_data = pd.DataFrame([{
    "Pneumonia": int(Pneumonia),
    "COPD": int(COPD),
    "age": age,
    "heartrate": heartrate,
    "SBP": SBP,
    "respiratoryrate": respiratoryrate,
    "spo2": spo2,
    "temperature": temperature,
    "WBC": WBC,
    "Albumin": Albumin,
    "ALT": ALT,
    "BUN": BUN,
    "sodium": sodium,
    "Plateletcount": Plateletcount,
    "SOFA": SOFA
}], columns=model.feature_names_in_)

# =====================================================
# 预测 + SHAP
# =====================================================
if st.button("🚨 Predict & Explain (SHAP)"):

    # ------------------ 预测 ------------------
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("## 🔍 Prediction Result")
    st.metric(
        label="Probability of Septic Shock Within 12 Hours",
        value=f"{prob:.2%}"
    )

    if prob >= 0.50:
        st.error("🚨 **High Risk**: Immediate evaluation and early aggressive intervention recommended.")
    elif prob >= 0.20:
        st.warning("⚠️ **Moderate Risk**: Close monitoring and early warning advised.")
    else:
        st.success("✅ **Low Risk**: Continue routine ICU care and observation.")

    st.markdown("---")

    # =================================================
    # SHAP 个体解释（终极稳定写法）
    # =================================================
    st.markdown("## 🧠 Individual Explanation (SHAP)")

    with st.spinner("Calculating SHAP values..."):
        explainer = shap.Explainer(model, feature_names=input_data.columns)
        shap_exp = explainer(input_data)

    # 🔥 关键：只取【第 0 个样本 + 正类（index=1）】
    shap_single = shap_exp[0, :, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_single, show=False)

    st.pyplot(fig)

# =====================================================
# 页脚
# =====================================================
st.markdown("""
---
**Disclaimer**  
This tool is intended for research and clinical decision support only.  
Final clinical decisions should be made by qualified healthcare professionals.
""")
