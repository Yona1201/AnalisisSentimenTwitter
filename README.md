# Analisis Sentimen Kepuasan Pelanggan Gojek

Proyek ini bertujuan menganalisis sentimen pelanggan Gojek menggunakan data dari aplikasi X. Kami menggunakan metode **SVM (Support Vector Machine)** untuk klasifikasi sentimen dan menyajikan hasil analisis dalam bentuk **wordcloud** dan **dashboard interaktif** berbasis Streamlit. Lihat dashboard di [sini](https://yona1201-analisissentimentwitter-app-x1jiig.streamlit.app/).

## Fitur

- **Pengumpulan Data** dari aplikasi X
- **Preprocessing**: pembersihan teks, stopwords, normalisasi
- **Labeling Sentimen**: positif, negatif, netral
- **Klasifikasi SVM**: model analisis sentimen
- **Visualisasi**: wordcloud dan grafik
- **Dashboard Streamlit**: visualisasi interaktif

## Teknologi

- **Python**, **Scikit-Learn**, **Streamlit**
- **Wordcloud**, **NLTK**, **Matplotlib**

## Cara Menjalankan

1. Clone repositori ini:
    ```bash
    git clone <URL repositori>
    cd <nama_folder_proyek>
    ```
2. Install dependensi:
    ```bash
    pip install -r requirements.txt
    ```
3. Jalankan aplikasi:
    ```bash
    streamlit run app.py
    ```

## Visualisasi

- **Wordcloud**: kata-kata sering muncul dalam ulasan
- **Grafik Sentimen**: proporsi sentimen pelanggan
