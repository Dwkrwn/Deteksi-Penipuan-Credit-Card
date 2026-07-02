import streamlit as st
import pandas as pd
import joblib
import time

# ==========================================================
# Load Model
# ==========================================================

model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ==========================================================
# Konfigurasi Halaman
# ==========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    
    layout="centered"
)

# ==========================================================
# Header
# ==========================================================

with st.sidebar:

    st.title("💳 Credit Card Fraud Detection")

    st.caption("Machine Learning Project")

    st.divider()

    menu = st.radio(
        "📂 Menu",
        [
            "🏠 Beranda",
            "📖 Panduan Penggunaan",
            "🔍 Prediksi Transaksi",
        ]
    )

    st.divider()

    st.subheader("Model")

    st.success("Random Forest")

    st.divider()

    st.subheader("Feature")

    st.write("✔ Skor Anomali Transaksi")

    st.write("✔ Indikator Risiko")

    st.write("✔ Skor Anomali Tambahan")

    st.write("✔ Sensor Pola Kecurangan")

    st.write("✔ Skor Integritas")

# ==========================================================
# Input User
# ==========================================================
if menu == "🏠 Beranda":

    st.title("💳 Credit Card Fraud Detection")

    st.subheader("Deteksi Penipuan Kartu Kredit")

    st.write("""
Selamat datang pada aplikasi **Credit Card Fraud Detection**.

Aplikasi ini dibangun menggunakan algoritma **Random Forest**
untuk memprediksi apakah suatu transaksi merupakan:

- ✅ Transaksi Normal

- 🚨 Transaksi Fraud

Silakan pilih menu **Prediksi Transaksi**
untuk mulai melakukan pengujian.
""")

elif menu == "📖 Panduan Penggunaan":

    st.title("📖 Panduan Penggunaan")

    st.info("""
Masukkan nilai dari lima indikator transaksi.

Rentang nilai yang disarankan:

-20.00 sampai 20.00
""")

    st.subheader("Penjelasan Indikator")

    st.markdown("""
- **Skor Anomali Transaksi** : Menggambarkan tingkat keanehan transaksi.

- **Indikator Risiko** : Menggambarkan tingkat risiko transaksi.

- **Skor Anomali Tambahan** : Memperkuat proses identifikasi.

- **Sensor Pola Kecurangan** : Mengidentifikasi pola fraud.

- **Skor Integritas** : Menggambarkan konsistensi transaksi.
""")
    st.subheader("🟢 Contoh Data Normal")

    st.table(pd.DataFrame({

    "Indikator":[
        "Skor Anomali",
        "Indikator Risiko",
        "Skor Anomali Tambahan",
        "Sensor Pola",
        "Skor Integritas"
    ],

    "Nilai":[
        -0.40,
        0.25,
        -0.30,
        0.42,
        -0.12
    ]

}))
    
    st.subheader("🔴 Contoh Data Fraud")

st.table(pd.DataFrame({

    "Indikator":[
        "Skor Anomali",
        "Indikator Risiko",
        "Skor Anomali Tambahan",
        "Sensor Pola",
        "Skor Integritas"
    ],

    "Nilai":[
        -8.75,
        -10.40,
        -7.60,
        -6.15,
        -5.70
    ]

}))

st.info("📌 Petunjuk Pengisian\n\nMasukkan nilai lima fitur PCA hasil transformasi dataset.\n\nRentang nilai yang disarankan adalah **-20.00 hingga 20.00**. Semakin sesuai dengan karakteristik data asli, semakin baik hasil prediksi model.")

st.subheader("Input Data Transaksi")

v14 = st.number_input(
    "Skor Anomali Transaksi (V14)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.01,
    format="%.2f",
    help="Menggambarkan tingkat keanehan suatu transaksi."
)

v17 = st.number_input(
    "Indikator Risiko Transaksi (V17)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.01,
    format="%.2f",
    help="Menunjukkan tingkat risiko transaksi berdasarkan pola historis."
)

v12 = st.number_input(
    "Skor Anomali Tambahan (V12)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.01,
    format="%.2f",
    help="Digunakan untuk memperkuat identifikasi transaksi yang tidak normal."
)

v10 = st.number_input(
    "Pola Kecurangan (V10)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.01,
    format="%.2f",
    help="Mewakili pola yang sering muncul pada transaksi fraud."
)

v16 = st.number_input(
    "Skor Integritas (V16)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.01,
    format="%.2f",
    help="Menggambarkan tingkat konsistensi atau integritas transaksi."
)

# ==========================================================
# Tombol Prediksi
# ==========================================================

if st.button("Prediksi"):
 
 with st.spinner("Sedang melakukan analisis..."):

    # Membuat DataFrame
    data = pd.DataFrame({

        "Faktor Anomali Transaksi A":[v14],
        "Indikator Utama Penyimpangan":[v17],
        "Faktor Anomali Transaksi B":[v12],
        "Sensor Pola Kecurangan":[v10],
        "Skor Integritas Transaksi":[v16]

    })

    # Scaling
    data = scaler.transform(data)

    # Prediksi
    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    st.divider()

    st.subheader("Hasil Prediksi")

    if prediction == 0:

        st.success("✅ Transaksi NORMAL")

        st.metric(
            label="Probabilitas Normal",
            value=f"{probability[0]*100:.2f}%"
        )

        st.success("""
Interpretasi

Model menilai transaksi
tidak menunjukkan pola
yang mengarah pada aktivitas penipuan.
""")

    else:

        st.error("🚨 TRANSAKSI FRAUD")

        st.metric(
            label="Probabilitas Fraud",
            value=f"{probability[1]*100:.2f}%"

        
        )

        st.warning("""
            Interpretasi

            Model mendeteksi pola transaksi
            yang memiliki kemiripan tinggi
            dengan transaksi penipuan.

            Disarankan dilakukan
            verifikasi sebelum transaksi diproses.
        """)

        
    st.metric( "Probabilitas", f"{probability[1]*100:.2f}%" )

    progress = st.progress(0)

    for i in range(100):

        time.sleep(0.01)

        progress.progress(i+1)


    # Footerr
st.divider()

st.caption(
"""
© 2026

Machine Learning Project

Universitas 17 Agustus 1945 Surabaya
"""
)