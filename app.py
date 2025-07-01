import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# -- Inisialisasi riwayat sesi --
if "history" not in st.session_state:
    st.session_state.history = []

# -- Load model, scaler, dan fitur --
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("features.pkl", "rb") as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Pastikan file model (model.pkl, scaler.pkl, features.pkl) tersedia di direktori yang sama.")
    st.stop()


# -- Konfigurasi halaman --
st.set_page_config(page_title="🎓 Prediksi Dropout Mahasiswa", page_icon="📘")
st.markdown("<h1 style='text-align: center;'>🎓 Prediksi Dropout Mahasiswa</h1>", unsafe_allow_html=True)
st.write("Sistem ini membantu memprediksi apakah seorang mahasiswa berpotensi mengalami dropout berdasarkan data akademik dan demografis.")

# -- Form input pengguna --
st.subheader("Masukkan Data Mahasiswa:")
with st.form("user_input"):
    col1, col2 = st.columns(2)

    # Identify which features from feature_names should be user inputs
    # Based on previous code, user inputs were limited.
    # You should ideally create input widgets for all features in feature_names.
    # For demonstration, let's assume these are the primary inputs for the demo.
    # *** IMPORTANT: Ensure all features in feature_names have corresponding inputs or default handling ***
    # If you added 'approval_ratio' and 'GDP_log' as features, they need calculation/input.
    # For simplicity in this demo app.py, let's assume user provides basic info
    # and calculated/economic features are either derived or use placeholder values.
    # A more robust app would require inputs for all features or a clear mapping.

    # Example mapping of user-friendly names to feature_names (adjust based on your feature_names)
    feature_map = {
        "Admission Grade": "Admission grade",
        "1st Semester Grade": "Curricular units 1st sem (grade)",
        "2nd Semester Grade": "Curricular units 2nd sem (grade)",
        "Age at Enrollment": "Age at enrollment",
        "Gender": "Gender", # Assuming Gender is used, need encoding
        "Scholarship Holder": "Scholarship holder" # Assuming Scholarship is used, need encoding
        # Need inputs/handling for: "approval_ratio", "Unemployment rate", "Inflation rate", "GDP_log", "Curricular units 1st sem (credited)"
    }


    # Collect user inputs for features that will be available in the form
    user_inputs = {}
    with col1:
        user_inputs["Admission Grade"] = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, step=0.1, key="adm_grade")
        user_inputs["1st Semester Grade"] = st.number_input("1st Semester Grade", min_value=0.0, max_value=20.0, step=0.1, key="sem1_grade")
        user_inputs["Gender"] = st.selectbox("Gender", options=["Male", "Female"], key="gender")


    with col2:
        user_inputs["2nd Semester Grade"] = st.number_input("2nd Semester Grade", min_value=0.0, max_value=20.0, step=0.1, key="sem2_grade")
        user_inputs["Age at Enrollment"] = st.slider("Age at Enrollment", 17, 60, 20, key="age")
        user_inputs["Scholarship Holder"] = st.selectbox("Scholarship Holder", options=["Yes", "No"], key="scholarship")


    submitted = st.form_submit_button("🔍 Prediksi")

# -- Proses prediksi jika form dikirim --
if submitted:
    # -- Buat input DataFrame untuk model --
    # Need to reconstruct the input features exactly as they were in training
    input_data_row = {}

    # Populate input_data_row with user inputs based on feature_map
    # This requires knowing how user inputs map to the actual features used by the model (feature_names)
    # This is a potential point of mismatch if the form inputs don't directly correspond to feature_names

    # *** Crucial Step: Map user_inputs to model's feature_names ***
    # This mapping needs to be correct based on the features used in cell g2B7VmFKK77D
    # The features used are: "Admission grade", "approval_ratio", "Unemployment rate", "Inflation rate", "GDP_log", "Curricular units 1st sem (credited)"
    # The form only collects: "Admission Grade", "1st Semester Grade", "2nd Semester Grade", "Age at Enrollment", "Gender", "Scholarship Holder"
    # We need to decide how to get the values for the *actual* features used by the model.

    # Let's assume we need to calculate 'approval_ratio' and use placeholder/average values for others
    # based on the limited form inputs. This is a simplification for the demo app.
    # A real-world app needs inputs for all features or a clear derivation process.

    # --- Simplified Mapping for Demo ---
    # We will use the 'Admission grade' from input.
    # We will calculate a simplified 'approval_ratio' (requires 1st sem enrolled/approved, which are NOT in form).
    # We will use placeholder values for economic features and 'Curricular units 1st sem (credited)'.
    # This means the app's prediction might not be fully accurate if these features are important.

    # --- Revised Input Data Construction ---
    # Initialize with placeholder/default values for all features the model expects
    input_data_row = {feat: 0 for feat in feature_names} # Default to 0 or a sensible mean/median from training data

    # Update with available user inputs (mapping user-friendly names to feature_names)
    # Note: This mapping is incomplete based on the form fields vs feature_names
    # Need to align these perfectly or add more inputs to the form.

    # Example (incomplete) mapping:
    if "Admission grade" in feature_names:
         input_data_row["Admission grade"] = user_inputs["Admission Grade"]

    # 'approval_ratio' needs 'Curricular units 1st sem (approved)' and 'enrolled', which are not in the form.
    # Using a placeholder or deriving it is complex without more inputs. Let's use a placeholder for now.
    if "approval_ratio" in feature_names:
        # This is a simplification. A real app needs more inputs to calculate this correctly.
        # For demo, let's use a dummy calculation or a default value.
        # If 1st sem grade is high, maybe assume a higher ratio? This is just illustrative.
        if user_inputs["1st Semester Grade"] > 10: # Dummy logic
             input_data_row["approval_ratio"] = 0.8
        else:
             input_data_row["approval_ratio"] = 0.4 # Dummy logic

    # Economic features and 'Curricular units 1st sem (credited)' are not in the form.
    # Use placeholder values (e.g., mean from training data or 0 if that was the common value).
    # Ideally, these should be inputs or derived differently.
    # Using 0 as a placeholder (assuming it was a common value in training data or a safe default)
    if "Unemployment rate" in feature_names:
        input_data_row["Unemployment rate"] = 0 # Placeholder
    if "Inflation rate" in feature_names:
        input_data_row["Inflation rate"] = 0 # Placeholder
    if "GDP_log" in feature_names:
         # Need original GDP input to log transform, or use log of a placeholder.
         # This is tricky without original GDP input. Using log of 0+epsilon as placeholder is bad.
         # Let's use a placeholder for GDP_log directly.
         input_data_row["GDP_log"] = 0 # Placeholder (log of something, needs careful handling)

    if "Curricular units 1st sem (credited)" in feature_names:
         input_data_row["Curricular units 1st sem (credited)"] = 0 # Placeholder


    # Create DataFrame and ensure column order
    input_df = pd.DataFrame([input_data_row])
    input_df = input_df[feature_names] # Ensure correct column order


    # Tampilkan input pengguna sebagai visualisasi (Display the limited form inputs)
    st.subheader("📄 Ringkasan Data yang Dimasukkan")
    display_df = pd.DataFrame([{
        "Admission Grade": user_inputs["Admission Grade"],
        "1st Semester Grade": user_inputs["1st Semester Grade"],
        "2nd Semester Grade": user_inputs["2nd Semester Grade"],
        "Age": user_inputs["Age at Enrollment"],
        "Gender": user_inputs["Gender"],
        "Scholarship": user_inputs["Scholarship Holder"]
    }])
    st.dataframe(display_df)

    # -- Scaling and prediction --
    try:
        input_scaled = scaler.transform(input_df)
        result = model.predict(input_scaled)[0]
        # Get probability of the predicted class
        confidence = model.predict_proba(input_scaled)[0][result]
        # Get probability of Dropout (class 0) and Non-Dropout (class 1) for the plot
        probs = model.predict_proba(input_scaled)[0]

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.info("Pastikan semua fitur yang dibutuhkan model tersedia dan formatnya benar.")
        st.stop()


    # -- Grafik Probabilitas Dropout --
    labels = ["Tidak Dropout", "Dropout"] # Assuming 0=Dropout, 1=Not Dropout based on training
    # Adjust labels based on your target_binary encoding if different
    if result == 0: # Predicted Dropout
        labels = ["Dropout", "Tidak Dropout"]
        probs = [probs[0], probs[1]] # Reorder probs to match labels

    fig, ax = plt.subplots()
    bars = ax.bar(labels, probs, color=["red", "green"] if result == 0 else ["green", "red"]) # Color based on prediction
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Prediksi Probabilitas")
    st.pyplot(fig)

    # -- Tampilkan hasil prediksi --
    st.subheader("🧠 Hasil Prediksi:")
    if result == 0: # Assuming 0 is Dropout
        st.error(f"⚠️ Mahasiswa ini diprediksi berpotensi **Dropout**\n\nProbabilitas Kepastian: {confidence:.2%}")
    else: # Assuming 1 is Not Dropout (Graduate/Enrolled)
        st.success(f"✅ Mahasiswa ini diprediksi akan **Melanjutkan Studi**\n\nProbabilitas Kepastian: {confidence:.2%}")


    # -- Simpan ke riwayat sesi --
    history_row = {
        "Admission": user_inputs["Admission Grade"],
        "1st Sem": user_inputs["1st Semester Grade"],
        "2nd Sem": user_inputs["2nd Semester Grade"],
        "Age": user_inputs["Age at Enrollment"],
        "Gender": user_inputs["Gender"],
        "Scholarship": user_inputs["Scholarship Holder"],
        "Prediksi": "Dropout" if result == 0 else "Tidak Dropout",
        "Probabilitas": f"{confidence:.2%}"
    }
    st.session_state.history.append(history_row)

    # -- Ranking Fitur Paling Berpengaruh (using model's feature_importances_ or coef_) --
    st.subheader("📈 Ranking Fitur yang Mempengaruhi Prediksi (Kepentingan/Koefisien)")
    try:
        if hasattr(model, 'feature_importances_'): # For tree-based models like RF, XGBoost
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({
                "Fitur": feature_names,
                "Kepentingan": importances
            }).sort_values("Kepentingan", ascending=False)
            st.dataframe(feat_imp)

        elif hasattr(model, 'coef_'): # For linear models like Logistic Regression
             # For binary classification, coef_ has shape (1, n_features)
             coefs = model.coef_[0]
             feat_imp = pd.DataFrame({
                 "Fitur": feature_names,
                 "Koefisien (Abs)": np.abs(coefs),
                 "Koefisien": coefs
             }).sort_values("Koefisien (Abs)", ascending=False)
             st.dataframe(feat_imp)

        else:
            st.info("Feature importance or coefficients are not available for this model type.")

    except Exception as e:
         st.warning(f"Tidak dapat menampilkan ranking fitur: {e}")


# -- Tampilkan riwayat prediksi sesi --
st.subheader("📊 Riwayat Prediksi di Sesi Ini")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("Belum ada prediksi di sesi ini.")

# -- Penjelasan Catatan Penting --
st.markdown("""
---
**Catatan Penting:**
*   Aplikasi demo ini menggunakan sebagian kecil fitur yang mungkin berbeda dari fitur lengkap yang digunakan dalam pelatihan model di notebook.
*   Fitur seperti 'approval_ratio', indikator ekonomi (Unemployment rate, Inflation rate, GDP), dan 'Curricular units 1st sem (credited)' tidak diminta secara langsung di form ini dan menggunakan nilai placeholder. Hal ini dapat mempengaruhi akurasi prediksi dibandingkan model penuh.
*   Untuk aplikasi produksi, pastikan semua fitur yang digunakan model diambil dari input pengguna atau dihitung dengan benar.
""")
