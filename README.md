# Laporan Proyek Machine Learning - Adil Faruq Habibi

## Domain Proyek
Domain proyek kali ini di bidang kesehatan yang dimana isu yang diangkat mengenai gagal jantung. Pendekatan machine learning terhadap isu ini adalah klasifikasi biner dimana model machine learning yang dibuat dapat memprediksi kemungkinan apakah seseorang mengalami gagal jantung atau tidak.

### Latar Belakang
Berdasarkan deskripsi dari sumber dataset [Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction), penyakit kardiovaskular (CVDs) adalah penyebab kematian nomor 1 secara global, dimana merenggut sekitar 17,9 juta jiwa setiap tahun, berkonstribusi sebanyak 31% dari semua kematian di seluruh dunia. Empat dari 5 CVD kematian disebabkan oleh serangan jantung dan stroke, dan sepertiga dari kematian ini terjadi pada orang di bawah usia 70 tahun. Gagal jantung adalah kejadian umum yang disebabkan oleh CVD. Orang dengan penyakit kardiovaskular atau yang berada pada risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia atau penyakit yang sudah ada) memerlukan deteksi dan manajemen dini di mana model machine learning dapat sangat membantu.

## Business Understanding
### Problem Statements
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:
- Bagaimana cara melakukan analisa fitur klinis dan menyeleksinya dari dataset agar dapat digunakan untuk membuat model yang baik?
- Bagaimana cara membangun model machine learning untuk mengklasifikasikan seseorang yang kemungkinan gagal jantung atau tidak?

### Goals
Tujuan dari proyek ini adalah sebagai berikut:
- Melakukan analisa fitur klinis dan menyeleksinya dari dataset agar menghasilkan model machine learning yang baik.
- Membangun beberapa model machine learning dengan evaluasi metrik dari model yang terpilih dengan akurasi yang tinggi dan less bias untuk mengklasifikasikan seseorang yang kemungkinan gagal jantung atau tidak dengan tepat.

### Solution statements
- Menggunakan beberapa algoritma diantaranya adalah KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier dan SupporVectorClassifier dengan melakukan fine-tuning hyperparameter untuk meminimalisir overfitting.
- Satu model yang nantinya digunakan untuk ujicoba prediksi berdasarkan tingkat akurasi tertinggi dari data uji yang less bias.

## Data Understanding

Data yang digunakan adalah Heart Failure Prediction Dataset dari Kaggle [disini](https://www.kaggle.com/fedesoriano/heart-failure-prediction).

  | Jenis                   | Keterangan                                                                              |
  | ----------------------- | --------------------------------------------------------------------------------------- |
  | Sumber                  | Dataset: [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction)          |
  | Dataset Owner           | fedesoriano                                                                             |
  | Lisensi                 | Open Database License (ODbL)                                                            |
  | Kategori                | health, heart condition, health condition, healthcare                                   |
  | Usability               | 10.0                                                                                    |
  | Jenis dan Ukuran Berkas | CSV (35.92 kB)                                                                          |
  | Jumlah Observasi        | 918 observations                                                                        |

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:

**Data Numerikal**
- Age: Umur pasien (tahun)
- RestingBP: Tekanan darah pasien yang sedang beristirahat (mmHg)
- Cholesterol: Kolesterol serum pasien (mg/dl)
- MaxHR: Detak jantung maksimum yang dicapai, berdetak per menit
- Oldpeak: Ukuran numerik dari depresi ST yang disebabkan oleh berolahraga relatif terhadap istirahat

**Data Kategorikal**
- Sex: Jenis kelamin pasien, laki-laki atau perempuan
- ChestPainType: Tipe nyeri data yang dialami pasien:
    - TA: Typical Angina
    - ATA: Atypical Angina
    - NAP: Non-Anginal Pain
    - ASY: Asymptomatic
- FastingBS: Gula darah pasien [1: jika glucose > 120 mg/dl dan 0: sebaliknya] 
- RestingECG: Hasil resting electrocardiogram:
    - Normal
    - ST: Memiliki kelainan gelombang ST-T (T inversi gelombang dan/atau ST elevasi atau depresi > 0.05 mV)
    - LVH: Menunjukkan kemungkinan atau pasti hipertensi ventrikel oleh Kriteria Etes
- ExerciseAngina: Angina yang diinduksi oleh olahraga, iya atau tidak
- ST_Slope: Gradien dari puncak latihan segmen ST
    - Up: Menanjak
    - Datar
    - Down: Menurun
- HeartDisease: 1: ya, 0: tidak (sebagai label)

**EDA**

1.  Distribusi data masing-masing kelas:
    
    ![kelas](https://drive.google.com/uc?export=view&id=1wPEFyDrp-dZt9ZUnlfvhMr_SOnX0IPbR)

2. Untuk fitur Kategorikal, berikut contoh eksplorasi dari fitur Sex: (untuk fitur lainnya bisa dilihat pada notebook)

    - Distribusi sebaran data pada laki-laki dan perempuan
    
        ![kelas](https://drive.google.com/uc?export=view&id=1FIDXMGtRqYF0PKdEt1efRIzQAwqIUv78)
        
        >Dari plot diatas terlihat kalau jumlah data pada laki-laki lebih banyak dibandingkan perempuan.
        
    - Proposisi data masing-masing kelas pada tiap fitur Sex
    
        ![kelas](https://drive.google.com/uc?export=view&id=1MjIJQlWiSwUbHYrS_sofZPdcF17yJBcu)
    
        > Dari plot diatas masing-masing gender lebih banyak termasuk kelas yang terkena gagal jantung
  
3. Untuk fitur numerikal, berikut contoh eksplorasi dari fitur Age: (untuk fitur lainnya bisa dilihat pada notebook)

    - Probabilitas dari setiap umur dan juga outlier dari masing-masing kelas
    
        ![kelas](https://drive.google.com/uc?export=view&id=19G7ulJPcGUa6fSKCIzxMWFbaINbA_Jmk)
        
        > Dari grafik diatas bisa dilihat bagaimana distribusi probabilitas pada rentang umur tiap bin tertentu pada masing-masing kelas, yang secara garis besar gagal jantung dialami oleh usia lanjut. Dan juga kita bisa simpulkan kalau penyakit gagal jantung pada usia dibawah 35 tahun adalah langka (dari outlier).

    - Korelasi antar fitur numerik dan ditambah fitur target
        
        ![kelas](https://drive.google.com/uc?export=view&id=18wOM4gQ7jBf0PCoNpMxO5y9K2K_8_RKk)
        
        > HeartDisease memiliki korelasi positif kuat terhadap OldPeak (korelasi = 0,4) dan korelasi negatif kuat untuk MaxHR (korelasi = -0,4). Dan untuk fitur lainnya bisa dikatakan berkorelasi moderat terhadap HeartDisease. Disamping itu juga ada hubungan korelasi negatif yang cukup kuat antara Age dan MaxHR sebesar -0,4. Jadi bisa disimpulkan dengan seiring bertambahnya usia, detak jantung cenderung menurun (lihat deskripsi fitur MaxHR).

## Data Preparation

1. Data Cleaning

    - Melakukan imputasi dengan strategy mean pada nilai 0 pada fitur Cholesterol. Alasannya berdasarkan pada informasi dari https://www.healthline.com/health/serum-cholesterol, ditemukan bahwa kandungan serum cholesterol tidak memungkinkan untuk bernilai 0 md/dL. Sehingga disimpulkan nilai 0 pada data adalah missing value.

    - Handling missing value, melakukan pembersihan pada data masing-masing fitur yang terdapat outlier dengan teknik Interquartile range / IQR. Jadi jumlah data yang awalnya berjumlah 918 menjadi 642.

2. Memisahkan kolom fitur dengan kolom target yang disimpan secara berturut-turut pada variable X dan y.

3. Melakkukan normalisasi pada data dengan rentang 0 hingga 1 dengan MinMaxScaler.

4. Dimensionality Reduction, dengan menggunakan principal component analysis / PCA. Dimana mula-mula mencari jumlah n_components yang optimal dilihat dari jumlah rasio variance yang secara signifikan tinggi dibandingkan yang lainnya. Lalu dari sana ditentukan ncomponents sebanyak 4 yang akan menjadi jumlah dimensi baru dari data fitur setelah PCA model dilatih dan ditransform.

5. Membagi data training dan testing dengan jumlah data testing sebanyak 20% dari keseluruhan data.

## Modeling

Pemodelan pada proyek ini menggunakan 4 algoritma machine learning yaitu K-Nearest Neighbor Classifier, Random Forest Classifier, AdaBoost Classifier dan Support Vector Classifier.

Pada masing-masing model dilakukan hyperparameter tuning untuk mendapatkan performa model yang optimal dan tidak overfitting. Fine-tuning pada masing-masing algoritma juga berbeda-beda. Fine-tuning pada setiap model, diantaranya : 
  - **KNN** dimana jumlah neighbors yang optimal adalah sebanyak 8. 
  - Pada **Random Forest** jumlah estimator diatur sebanyak 70, max depth 16 dan max leaf nodes 20. Semakin besar maksimum kedalaman semakin model menjadi cendrung overfitting. 
  - Lalu pada model **AdaBoosting** tidak dilakukan tuning karena sudah cukup bagus dengan model yang sederhana saja. 
  - Dan pada **SVC** menggunakan kernel sigmoid dengan argumen gamma diset menjadi auto.

### Model Selection

Indikator dalam memilih model yang akan jadi model utama adalah:
  
  - **Akurasi** dari data test yang di-fed kepada model paling tinggi
  - **Generalisasi**, artinya bias model terhadap data baru yang diprediksikan tidak besar. Dan model yang dipilih tidak overfitting.

Berikut f1-score pada masing-masing model dari hasil evaluasi terhadap data train dan test:

![kelas](https://drive.google.com/uc?export=view&id=1DoDLdCngfci6x6j2my5dhCoVDB6KGRWT)
    
![kelas](https://drive.google.com/uc?export=view&id=1Lt2Hj-ZxoyPap09dSUa_zKyKcEoFk0hN)

Berdasarkan indikator memilih model, dari dataframe dan bar plot diatas model yang dipilih adalah **AdaBoostingClassifier**. Hal ini dikarenakan f1-score pada model ini dan RandomForestClassifier yang cukup identik tap bias prediktif train dan test lebih kecil meskipun RandomForest memiliki f1-score yang lebih tinggi.

## Evaluation

Metrik evaluasi yang digunakan adalah Confusion matrix dan Classification report.

1. Confusion Matrix
    
    ![kelas](https://drive.google.com/uc?export=view&id=1IAvE6PBd3tZrUCMV6V6zbKsNeFtg8BAm)
    
    > Dari data test dilakukan prediksi dengan model AdaBoosting lalu hasilnya yang dibandingkan dengan nilai aktual dan disajikan pada pada matrik diatas. Dimana untuk kasus klasifikasi gagal jantung ini perlu perhatian lebih. Dimana dari hasil prediksi terdapat 8 orang pasien yang diprediksi tidak sakit padahal sebenarnya sakit. Still need some improvement!

2. Classification Report

    ![kelas](https://drive.google.com/uc?export=view&id=1aCpcuoAhsaFcm9VkrvG6r6h_hrUeSkks)
    
  - Dari precision, persentase model berhasil memprediksi kasus positif / heart disease dari seluruh kasus positif yang diprediksi adalah 89%. Dan 86% untuk kasus sebaliknya
  - Dari recall, persentase model berhasil memprediksi kasus positif / heart disease dengan benar dari seluruh kasus positif aktul adalah 89%. dan 86% untuk kasus sebaliknya.
  
    ![kelas](https://drive.google.com/uc?export=view&id=1B83ql2B8nJbW-THENOptOkJV77P8_i6S)
  
  - Sedangkan f1 score adalah harmonic mean dari precision dan recall. Ini juga digunakan untuk kasus unbalanced data karena memperhitungkan FP dan FN.
  
    ![kelas](https://drive.google.com/uc?export=view&id=1c8QYyHsMvcCmnlegXXO08sIestrx5lmE)
