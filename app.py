import streamlit as st
import os
from src.predictor import predict_glaucoma

st.set_page_config(page_title="Glaucoma Detector", page_icon="👁️", layout="centered")
st.title("👁️ Glaucoma Eye Disease Detector")

st.markdown("Upload a retinal image to detect glaucoma.")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(temp_path, caption="Uploaded Image", width=300)

    with st.spinner("Analyzing..."):
        result = predict_glaucoma(temp_path)

    if result == "Healthy":
        st.success("✅ Eye appears Healthy")
        st.balloons()
    else:
        st.error("⚠️ Glaucoma Detected. Please consult a doctor.")

    os.remove(temp_path)

st.caption("⚠️ Educational project. Not a medical diagnosis.")
