# 🎯 Overview Proyek

## Latar Belakang
Proyek di dalam repositori **Learning-Data-Science-2026** ini dibuat sebagai wadah eksperimen komprehensif untuk menyelesaikan permasalahan estimasi nilai kontinu (**Regresi**). Fokus utama dari proyek ini bukan sekadar mencari model dengan akurasi tertinggi, melainkan berfokus pada **standardisasi penulisan kode** menggunakan Machine Learning Pipeline yang bersih, *scalable*, dan bebas dari kebocoran data (*data leakage*).

## Tujuan Proyek
1. **Automasi Preprocessing:** Membangun alur pembersihan data yang konsisten untuk tipe data numerik maupun kategorikal.
2. **Efisiensi Fitur:** Mengimplementasikan seleksi fitur otomatis secara maju (*Forward Sequential Feature Selection*) untuk mereduksi dimensi data yang tidak berkontribusi signifikan.
3. **Komparasi Algoritma:** Menguji keandalan berbagai algoritma berbasis *Tree-based Ensemble* (Bagging vs Boosting).
4. **Optimasi Model:** Menemukan kombinasi hyperparameter terbaik secara efisien menggunakan pencarian acak (`RandomizedSearchCV`).

## Metrik Evaluasi
Model dievaluasi menggunakan metrik **$R^2$ Score (Coefficient of Determination)** untuk mengukur seberapa baik variasi pada variabel dependen dapat dijelaskan oleh fitur-fitur independen di dalam model. Target eksperimen adalah mencapai nilai $R^2$ yang stabil baik pada data latih (*train*) maupun data uji (*test*).
