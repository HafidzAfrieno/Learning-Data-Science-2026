# 🛠 Arsitektur & Alur Kerja Pipeline

Proyek ini mengadopsi arsitektur *end-to-end* menggunakan `sklearn.pipeline.Pipeline`. Seluruh proses transformasi data diikat menjadi satu kesatuan unit estimator untuk memastikan proses `fit` dan `transform` berjalan secara terisolasi pada setiap *fold* Cross-Validation.

## Detail Alur Kerja

### 1. Preprocessing (`preprocessor`)
Tahap awal untuk mempersiapkan data mentah sebelum masuk ke model:
* **Fitur Numerik:** Penanganan nilai kosong menggunakan Imputer (Median/Mean) diikuti dengan penskalaan nilai via `StandardScaler` atau `MinMaxScaler`.
* **Fitur Kategorikal:** Penanganan nilai kosong menggunakan Imputer (Modus/Konstan) diikuti dengan konversi teks ke angka via `OneHotEncoder` (dengan opsi `handle_unknown='ignore'`).

### 2. Feature Selection (`SequentialFeatureSelector`)
Menggunakan pendekatan *Wrapper Method* berupa **Forward Selection**:
* Algoritma memulai dengan 0 fitur, lalu menguji fitur satu per satu.
* Fitur yang memberikan peningkatan performa paling signifikan pada estimator `DecisionTreeRegressor(max_depth=2)` akan dipilih terlebih dahulu.
* Proses berhenti secara kokoh ketika jumlah fitur yang diinginkan (`n_features_to_select`) terpenuhi (diuji pada variasi 4 dan 5 fitur).

### 3. Model Regression (`model_regression`)
Tahap akhir di mana data yang telah bersih dan ringkas disuapkan ke algoritma Machine Learning utama untuk melatih fungsi prediksi.
