# 🎓 Prediksi Dropout Mahasiswa

Aplikasi Streamlit untuk memprediksi risiko dropout mahasiswa berbasis data akademik dan demografis.

## Fitur Utama
- Prediksi dropout menggunakan model machine learning.
- Visualisasi probabilitas prediksi (Dropout vs Tidak Dropout).
- Riwayat prediksi pada sesi saat ini.
- Ranking fitur paling berpengaruh (simulasi SHAP sederhana).

## Cara Menjalankan
1. Pastikan Python 3.8+ sudah terpasang.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
4. Buka browser ke alamat yang tertera (biasanya http://localhost:8501).

## File Penting
- `app.py` : Source code aplikasi Streamlit
- `model.pkl` : Model machine learning terlatih
- `scaler.pkl` : Scaler (normalisasi fitur)
- `features.pkl` : Daftar urutan fitur yang digunakan model

## Catatan
- Model, scaler, dan fitur harus sesuai dengan data training.
- Untuk interpretasi fitur, aplikasi menggunakan simulasi SHAP berbasis koefisien model Logistic Regression.
