import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# -- Inisialisasi riwayat sesi --
if "history" not in st.session_state:
    st.session_state.history = []

# -- Load model, scaler, dan fitur --
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# -- Konfigurasi halaman --
st.set_page_config(page_title="🎓 Prediksi Dropout Mahasiswa", page_icon="📘")
st.markdown("<h1 style='text-align: center;'>🎓 Prediksi Dropout Mahasiswa</h1>", unsafe_allow_html=True)
st.write("Sistem ini membantu memprediksi apakah seorang mahasiswa berpotensi mengalami dropout berdasarkan data akademik dan demografis.")

# -- Form input pengguna --
with st.form("user_input"):
    col1, col2 = st.columns(2)

    with col1:
        admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, step=0.1)
        sem1_grade = st.number_input("1st Semester Grade", min_value=0.0, max_value=20.0, step=0.1)
        gender = st.selectbox("Gender", options=["Male", "Female"])
    
    with col2:
        sem2_grade = st.number_input("2nd Semester Grade", min_value=0.0, max_value=20.0, step=0.1)
        age = st.slider("Age at Enrollment", 17, 60, 20)
        scholarship = st.selectbox("Scholarship Holder", options=["Yes", "No"])

    submitted = st.form_submit_button("🔍 Prediksi")

# -- Proses prediksi jika form dikirim --
if submitted:
    # -- Buat input awal --
    input_data = {
        "Admission grade": admission_grade,
        "Curricular units 1st sem (grade)": sem1_grade,
        "Curricular units 2nd sem (grade)": sem2_grade,
        "Age at enrollment": age,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Scholarship holder_Yes": 1 if scholarship == "Yes" else 0
    }

    # -- Buat DataFrame dari input dan lengkapi fitur --
    input_df = pd.DataFrame([input_data])

    # Tambahkan kolom-kolom yang belum ada (isi default = 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Susun urutan kolom
    input_df = input_df[feature_names]

    # Tampilkan input pengguna sebagai visualisasi
    st.subheader("📄 Ringkasan Data yang Dimasukkan")
    display_df = pd.DataFrame([{
        "Admission Grade": admission_grade,
        "1st Semester Grade": sem1_grade,
        "2nd Semester Grade": sem2_grade,
        "Age": age,
        "Gender": gender,
        "Scholarship": scholarship
    }])
    st.dataframe(display_df)

    # -- Scaling dan prediksi --
    input_scaled = scaler.transform(input_df)
    result = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][result]

    # -- Grafik Probabilitas Dropout --
    labels = ["Tidak Dropout", "Dropout"]
    probs = model.predict_proba(input_scaled)[0]
    fig, ax = plt.subplots()
    ax.bar(labels, probs, color=["green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Prediksi Dropout")
    st.pyplot(fig)

    # -- Tampilkan hasil prediksi --
    st.subheader("🧠 Hasil Prediksi:")
    if result == 1:
        st.error(f"⚠️ Mahasiswa ini diprediksi berpotensi dropout\n\nProbabilitas: {confidence:.2%}")
    else:
        st.success(f"✅ Mahasiswa ini diprediksi akan melanjutkan studi\n\nProbabilitas: {confidence:.2%}")

    # -- Simpan ke riwayat sesi --
    history_row = {
        "Admission": admission_grade,
        "1st Sem": sem1_grade,
        "2nd Sem": sem2_grade,
        "Age": age,
        "Gender": gender,
        "Scholarship": scholarship,
        "Prediksi": "Dropout" if result == 1 else "Tidak Dropout",
        "Probabilitas": f"{confidence:.2%}"
    }
    st.session_state.history.append(history_row)

    # -- Ranking Fitur Paling Berpengaruh (simulasi SHAP) --
    coefs = model.coef_[0]
    contributions = input_scaled[0] * coefs
    feat_imp = pd.DataFrame({
        "Fitur": feature_names,
        "Kontribusi": contributions
    }).sort_values("Kontribusi", key=abs, ascending=False)

    st.subheader("📈 Ranking Fitur yang Mempengaruhi Prediksi")
    st.dataframe(feat_imp.head(10))

# -- Tampilkan riwayat prediksi sesi --
st.subheader("📊 Riwayat Prediksi di Sesi Ini")
st.dataframe(pd.DataFrame(st.session_state.history))
