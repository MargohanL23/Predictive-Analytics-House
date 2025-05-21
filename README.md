# Laporan Proyek Machine Learning - [MARGOHAN L. SIRINGO-RINGO]

## Project Overview

Pasar properti merupakan salah satu sektor ekonomi yang paling dinamis dan kompleks. Fluktuasi harga properti memiliki dampak signifikan terhadap individu, bisnis, dan bahkan stabilitas ekonomi makro. Bagi calon pembeli, memahami faktor-faktor yang memengaruhi harga rumah sangat penting untuk membuat keputusan investasi yang tepat. Demikian pula, bagi penjual, penetapan harga yang kompetitif dapat mempercepat proses penjualan dan memaksimalkan keuntungan. Di sisi lain, lembaga keuangan yang terlibat dalam pinjaman hipotek juga sangat bergantung pada penilaian properti yang akurat.

Secara tradisional, penilaian properti seringkali melibatkan inspeksi fisik dan analisis pasar oleh penilai profesional. Namun, metode ini bisa memakan waktu, mahal, dan terkadang subjektif. Dengan ledakan data besar dan kemajuan dalam ilmu data, terdapat peluang besar untuk memanfaatkan teknik **Machine Learning** dalam memprediksi harga rumah secara lebih efisien dan akurat. Model prediksi harga rumah berbasis machine learning dapat menganalisis berbagai fitur properti dan tren pasar untuk memberikan estimasi harga yang lebih objektif dan cepat. Ini dapat membantu berbagai pihak dalam proses jual beli properti, mulai dari individu yang mencari tempat tinggal, investor yang ingin berinvestasi, hingga pengembang properti yang merencanakan proyek baru.

Proyek ini bertujuan untuk membangun model predictive analytics menggunakan teknik **regresi** untuk memprediksi harga rumah berdasarkan karakteristik properti. Solusi ini diharapkan dapat memberikan estimasi harga yang lebih cepat, akurat, dan dapat diandalkan dibandingkan metode konvensional, sehingga dapat menjadi alat bantu yang berharga di pasar properti.

### Referensi Tambahan:
* [Contoh penerapan Machine Learning dalam real estate](https://medium.com/@sanaalam/machine-learning-in-real-estate-c23f4625b03e) (Medium.com, Sana Alam, 2021)
* [Pentingnya data dalam prediksi harga properti](https://www.forbes.com/sites/forbesrealestatecouncil/2023/07/25/the-power-of-data-in-real-estate/?sh=355e105e6080) (Forbes, Forbes Real Estate Council, 2023)

---

## Business Understanding

Proses klarifikasi masalah dalam proyek prediksi harga rumah ini melibatkan identifikasi kebutuhan pengguna dan tujuan bisnis yang ingin dicapai.

### Problem Statements

1.  **Ketidakpastian dalam Penentuan Harga Jual/Beli:** Calon pembeli dan penjual sering menghadapi kesulitan dalam menentukan harga properti yang wajar dan kompetitif di pasar, menyebabkan proses tawar-menawar yang panjang atau estimasi yang tidak akurat.
2.  **Keterbatasan Penilaian Properti Tradisional:** Penilaian properti secara manual oleh profesional bisa memakan waktu dan biaya, serta berpotensi adanya bias subjektif, terutama di pasar yang bergerak cepat.
3.  **Kurangnya Alat Bantu Prediktif yang Efisien:** Individu atau entitas yang ingin memprediksi nilai properti di masa depan atau mengidentifikasi properti dengan potensi apresiasi nilai sering kali kekurangan alat yang efisien dan berbasis data untuk melakukannya.

### Goals

1.  Membangun model regresi yang mampu **memprediksi nilai median rumah** berdasarkan fitur-fitur properti yang tersedia dalam dataset.
2.  Menyediakan estimasi nilai properti yang **lebih cepat dan objektif**, mengurangi ketergantungan pada penilaian manual yang memakan waktu dan biaya.
3.  Mengidentifikasi **faktor-faktor kunci** yang paling berpengaruh terhadap nilai rumah, memberikan wawasan berharga bagi pembeli, penjual, dan pengembang properti.

### Solution Approach

Untuk mencapai tujuan di atas, kami akan menggunakan pendekatan machine learning dengan membangun model regresi.

#### Solution Statements

1.  **Pendekatan Algoritma Regresi Linier (Linear Regression):** Model ini akan digunakan sebagai *baseline* karena kesederhanaan dan interpretasinya yang mudah. Model ini akan memodelkan hubungan linier antara fitur-fitur input dan nilai median rumah.
2.  **Pendekatan Algoritma Ensemble (Random Forest Regressor / Gradient Boosting Regressor):** Untuk potensi akurasi yang lebih tinggi dan kemampuan menangani hubungan non-linier serta interaksi fitur, model *ensemble* seperti Random Forest Regressor atau Gradient Boosting Regressor akan dipertimbangkan. Algoritma ini menggabungkan prediksi dari beberapa *decision tree* untuk menghasilkan prediksi yang lebih robust.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Prediksi Harga Rumah (House Price Prediction)**. Dataset ini bersumber dari Kaggle, dengan nama file `housing.csv`. Dataset ini berisi informasi properti dan harga jual rumah dari ribuan sampel, memenuhi kriteria minimum 500 sampel data kuantitatif yang dibutuhkan.

**Sumber Dataset:** [https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction/input?select=housing.csv](https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction/input?select=housing.csv)

**Informasi Umum Data:**
Dataset ini memiliki 20.640 baris (sampel) dan 10 kolom (fitur). Sembilan dari sepuluh kolom adalah numerik (`float64`), dan satu kolom (`ocean_proximity`) adalah kategorikal (`object`). Ukuran memori yang digunakan oleh dataset adalah sekitar 1.6+ MB.

**Detail Kolom:**

| # | Column              | Non-Null Count | Dtype   | Deskripsi                                                                |
| :- | :------------------ | :------------- | :------ | :----------------------------------------------------------------------- |
| 0 | `longitude`         | 20640          | `float64` | Posisi geografis bujur untuk blok perumahan.                             |
| 1 | `latitude`          | 20640          | `float64` | Posisi geografis lintang untuk blok perumahan.                           |
| 2 | `housing_median_age`| 20640          | `float64` | Usia median rumah dalam blok perumahan tersebut.                        |
| 3 | `total_rooms`       | 20640          | `float64` | Jumlah total ruangan di semua rumah dalam blok perumahan.                |
| 4 | `total_bedrooms`    | 20433          | `float64` | Jumlah total kamar tidur di semua rumah dalam blok perumahan.            |
| 5 | `population`        | 20640          | `float64` | Jumlah populasi di blok perumahan.                                       |
| 6 | `households`        | 20640          | `float64` | Jumlah total rumah tangga, yaitu jumlah rumah dalam blok perumahan.      |
| 7 | `median_income`     | 20640          | `float64` | Pendapatan median untuk rumah tangga dalam blok perumahan (dalam puluhan ribu USD). |
| 8 | `median_house_value`| 20640          | `float64` | Nilai median rumah untuk rumah tangga dalam blok perumahan (variabel target). |
| 9 | `ocean_proximity`   | 20640          | `object`  | Lokasi relatif blok perumahan terhadap samudra/laut.                     |

**Missing Values:**
Hanya kolom `total_bedrooms` yang memiliki nilai hilang sebanyak 207 dari 20640 sampel (sekitar 1.00%).

### Exploratory Data Analysis (EDA) dan Insight

#### 1. Analisis Missing Values:
* Ditemukan 207 nilai hilang pada kolom `total_bedrooms`. Persentase nilai hilang ini sekitar 1.00% dari total dataset.

#### 2. Deteksi Outlier (Metode IQR):
Analisis *outlier* menggunakan metode Interquartile Range (IQR) pada kolom numerik menunjukkan keberadaan *outlier* yang signifikan pada beberapa fitur kunci dan variabel target:

* `longitude`: 0 outliers (0.00%)
* `latitude`: 0 outliers (0.00%)
* `housing_median_age`: 0 outliers (0.00%)
* `total_rooms`: 1287 outliers (6.24%)
* `total_bedrooms`: 1306 outliers (6.33%)
* `population`: 1196 outliers (5.79%)
* `households`: 1220 outliers (5.91%)
* `median_income`: 681 outliers (3.30%)
* `median_house_value`: 1071 outliers (5.19%)

Keberadaan *outlier* ini, terutama pada kolom-kolom seperti `total_rooms`, `total_bedrooms`, `population`, `households`, dan variabel target `median_house_value` (yang semuanya memiliki lebih dari 5% *outlier*), menunjukkan bahwa kolom-kolom ini kemungkinan besar memiliki distribusi yang **skewed** (cenderung ke satu sisi) dan nilai-nilai ekstrem. Ini memerlukan penanganan khusus pada tahap *data preparation*.

#### 3. Korelasi Variabel Numerik:
Visualisasi *heatmap* korelasi antar variabel numerik memberikan insight penting:

- median_income menunjukkan korelasi positif yang sangat kuat (0.69) dengan median_house_value. Ini mengindikasikan bahwa pendapatan median di suatu area adalah prediktor paling dominan untuk nilai median rumah.
- Fitur-fitur seperti total_rooms, total_bedrooms, population, dan households menunjukkan korelasi yang sangat tinggi satu sama lain (semuanya di atas 0.85, bahkan mencapai 0.93 antara total_rooms dan total_bedrooms). Ini adalah indikasi multicollinearity yang signifikan. Meskipun model ensemble seperti Random Forest cenderung robust terhadap multicollinearity, ini dapat memengaruhi interpretasi koefisien pada model Regresi Linier.
- longitude dan latitude juga menunjukkan korelasi negatif yang sangat kuat (-0.92), yang wajar mengingat sifat koordinat geografis di wilayah tertentu. Korelasi mereka dengan median_house_value relatif lemah, menunjukkan bahwa dampak posisi geografis mungkin lebih kompleks daripada hubungan linier sederhana.
- Fitur total_bedrooms, population, dan households menunjukkan korelasi yang sangat lemah atau bahkan sedikit negatif dengan median_house_value. Ini mungkin mengindikasikan bahwa jumlah absolut dari fitur-fitur ini kurang relevan dibandingkan rasio-rasio tertentu (misalnya, kamar per rumah tangga) atau bahwa korelasi mereka tertutupi oleh variabel lain.

---

## Data Preparation

Tahap persiapan data sangat krusial untuk memastikan kualitas data yang masuk ke model machine learning dan mengoptimalkan performanya. Berikut adalah teknik-teknik data preparation yang akan dilakukan secara berurutan:

1.  **Penanganan *Missing Values*:**
    * **Proses:** Untuk kolom `total_bedrooms` yang memiliki 207 nilai hilang, kami akan mengisi nilai-nilai ini dengan **median** dari kolom `total_bedrooms` yang ada.
    * **Alasan:** Menggunakan median adalah pilihan yang robust karena kurang sensitif terhadap *outlier* dibandingkan rata-rata (mean), sehingga membantu menjaga distribusi asli data dan memberikan imputasi yang lebih representatif untuk kolom `total_bedrooms`. Imputasi ini penting untuk memastikan semua data numerik lengkap dan dapat diproses oleh model.

2.  **Penanganan *Outlier* dan Normalisasi Distribusi (Transformasi Logaritmik):**
    * **Proses:** Berdasarkan analisis EDA, beberapa fitur numerik (`total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`) dan variabel target `median_house_value` menunjukkan distribusi yang **skewed** dan keberadaan *outlier* yang signifikan. Kami akan menerapkan **transformasi logaritmik** (`np.log1p`) pada kolom-kolom ini untuk mengurangi *skewness* dan dampak *outlier*.
    * **Alasan:** Transformasi logaritmik membantu menormalkan distribusi data yang *skewed* dan memadatkan rentang nilai ekstrem, membuat data lebih sesuai untuk model regresi linier yang sensitif terhadap asumsi normalitas dan homoskedastisitas. Selain itu, transformasi ini membantu model *ensemble* seperti Random Forest belajar dari pola data yang lebih konsisten tanpa menghilangkan informasi berharga yang terkandung dalam nilai-nilai ekstrem. Untuk variabel target, hasil prediksi logaritmik akan di-*inverse transform* menggunakan `np.expm1` untuk mendapatkan nilai dalam skala asli.

3.  **Encoding Fitur Kategorikal:**
    * **Proses:** Fitur kategorikal `ocean_proximity` perlu diubah menjadi representasi numerik. Karena fitur ini tidak memiliki urutan intrinsik dan memiliki beberapa kategori unik, **One-Hot Encoding** akan digunakan untuk membuat kolom biner baru (0 atau 1) untuk setiap kategori unik.
    * **Alasan:** Sebagian besar algoritma machine learning hanya dapat bekerja dengan data numerik. One-Hot Encoding memungkinkan fitur kategorikal untuk diintegrasikan ke dalam proses pemodelan tanpa mengasumsikan hubungan ordinal yang salah.

4.  **Feature Scaling:**
    * **Proses:** Fitur-fitur numerik yang memiliki skala berbeda (termasuk `longitude`, `latitude`, `housing_median_age`, dan fitur-fitur yang telah ditransformasi logaritmik) akan disesuaikan skalanya menggunakan metode seperti **StandardScaler**.
    * **Alasan:** Banyak algoritma machine learning, terutama yang berbasis jarak (misalnya k-Nearest Neighbors) atau optimasi gradien (misalnya Regresi Linier dan algoritma berbasis *neural network*), sangat sensitif terhadap skala fitur. *Scaling* membantu mencegah fitur dengan skala besar mendominasi proses pelatihan dan mempercepat konvergensi model.

---

## Modeling

Tahapan ini membahas mengenai pembangunan model regresi yang Anda buat untuk menyelesaikan permasalahan prediksi nilai median rumah. Kami akan menyajikan dua solusi dengan algoritma yang berbeda.

1.  **Linear Regression:**
    * **Penjelasan:** Regresi Linier adalah algoritma *supervised learning* yang memodelkan hubungan linier antara variabel dependen (target, yaitu `median_house_value`) dan satu atau lebih variabel independen (fitur). Model ini berusaha menemukan koefisien terbaik untuk setiap fitur untuk meminimalkan jumlah kuadrat residu (perbedaan antara nilai aktual dan prediksi).
    * **Kelebihan:**
        * **Sederhana dan Interpretatif:** Mudah dipahami dan diinterpretasikan. Koefisien model menunjukkan pengaruh masing-masing fitur terhadap nilai rumah.
        * **Cepat Dilatih:** Proses pelatihan yang relatif cepat, bahkan pada dataset besar.
    * **Kekurangan:**
        * **Asumsi Linieritas:** Membutuhkan hubungan linier antara fitur dan target. Meskipun telah dilakukan transformasi logaritmik, hubungan ini mungkin tidak sepenuhnya linier.
        * **Sensitif terhadap *Outlier*:** Meskipun telah ditangani dengan transformasi, sisa *outlier* atau data yang tidak sempurna bisa tetap memengaruhi.
        * **Sensitif terhadap Multicollinearity:** Adanya *multicollinearity* yang tinggi antar fitur (seperti yang terdeteksi) dapat menyebabkan koefisien model menjadi tidak stabil dan sulit diinterpretasikan, meskipun model tetap bisa membuat prediksi.

2.  **Random Forest Regressor:**
    * **Penjelasan:** Random Forest adalah algoritma *ensemble learning* yang membangun banyak *decision tree* selama pelatihan dan mengeluarkan prediksi rata-rata dari masing-masing pohon untuk masalah regresi. Ini adalah algoritma yang kuat dan fleksibel yang dapat menangani data non-linier dan interaksi fitur.
    * **Kelebihan:**
        * **Akurasi Tinggi:** Seringkali memberikan akurasi yang sangat baik karena menggabungkan kekuatan banyak pohon, mengurangi varians.
        * **Tidak Rentan Terhadap *Overfitting*:** Kurang cenderung *overfit* dibandingkan *single decision tree* karena teknik *random sampling* pada fitur dan data (bagging).
        * **Menangani Non-Linieritas dan Interaksi Fitur:** Mampu menangani hubungan yang kompleks dan non-linier antara fitur dan target tanpa asumsi spesifik.
        * **Robust Terhadap *Outlier* dan Multicollinearity:** Lebih tangguh terhadap *outlier* dan *multicollinearity* dibandingkan Regresi Linier, menjadikannya pilihan yang sangat baik untuk dataset ini.
    * **Kekurangan:**
        * **Kurang Interpretatif:** Model ini dianggap "kotak hitam" karena sulit untuk menginterpretasikan bagaimana setiap fitur memengaruhi prediksi secara individual dibandingkan Regresi Linier.
        * **Waktu Pelatihan Lebih Lama:** Membutuhkan waktu pelatihan yang lebih lama dan lebih banyak sumber daya komputasi dibandingkan Regresi Linier, meskipun `n_jobs=-1` membantu mempercepatnya.

---

## Evaluation

Pada tahap ini, kami mengevaluasi kinerja kedua model regresi yang telah dilatih menggunakan metrik yang relevan untuk masalah prediksi nilai: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R-squared ($R^2$). Metrik ini dihitung berdasarkan perbandingan antara nilai `median_house_value` aktual (`y_test_original_scale`) dan nilai prediksi dari masing-masing model (`y_pred_lr_original_scale` dan `y_pred_rf_original_scale`), yang semuanya telah dikembalikan ke skala asli.

1.  **Mean Squared Error (MSE):**
    * **Formula:** $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
    * **Bagaimana Bekerja:** MSE mengukur rata-rata dari kuadrat perbedaan antara nilai aktual ($y_i$) dan nilai prediksi ($\hat{y}_i$). Karena kesalahan dikuadratkan, MSE sangat sensitif terhadap *outlier* atau kesalahan prediksi yang besar. Nilai MSE yang lebih rendah menunjukkan akurasi yang lebih baik.

2.  **Root Mean Squared Error (RMSE):**
    * **Formula:** $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
    * **Bagaimana Bekerja:** RMSE adalah akar kuadrat dari MSE. Metrik ini lebih mudah diinterpretasikan karena memiliki unit yang sama dengan variabel target (nilai rumah). Nilai yang lebih rendah menunjukkan akurasi yang lebih baik.

3.  **R-squared ($R^2$):**
    * **Formula:** $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
    * **Bagaimana Bekerja:** $R^2$ menunjukkan proporsi varians dalam variabel dependen yang dapat dijelaskan oleh model. Nilai $R^2$ berkisar antara 0 hingga 1. Nilai 1 menunjukkan model yang sangat baik, sedangkan nilai yang lebih rendah menunjukkan model yang kurang efektif dalam menjelaskan varians target.

**Hasil Proyek Berdasarkan Metrik Evaluasi:**

Setelah melatih dan mengevaluasi kedua model pada *test set*, berikut adalah hasilnya:

| Model                 | MSE          | RMSE       | R-squared ($R^2$) |
| :-------------------- | :----------- | :--------- | :---------------- |
| Linear Regression     | 5285759704.38 | 72703.23  | 0.60             |
| Random Forest Regressor | 2453584613.69 | 49533.67  | 0.81             |

**Analisis Hasil:**

* Model **Random Forest Regressor** menunjukkan performa yang secara signifikan lebih unggul dibandingkan **Linear Regression**. Hal ini terlihat dari nilai **RMSE yang jauh lebih rendah (49533.67 vs 72703.23)** dan nilai **$R^2$ yang jauh lebih tinggi (0.81 vs 0.60)**.
* **RMSE sebesar 49533.67** untuk Random Forest Regressor berarti rata-rata kesalahan prediksi nilai rumah median adalah sekitar $49.533,67. Ini menunjukkan tingkat akurasi yang baik dalam memprediksi nilai properti.
* **$R^2$ sebesar 0.81** untuk Random Forest Regressor menunjukkan bahwa sekitar 81% variabilitas nilai rumah median dapat dijelaskan oleh fitur-fitur yang dimasukkan dalam model. Angka ini sangat baik dan mengindikasikan model yang kuat.
* Meskipun Linear Regression telah diuntungkan oleh transformasi logaritmik dan *feature scaling*, performa prediktifnya tetap tidak sebaik Random Forest. Hal ini mungkin karena hubungan antara fitur properti dan nilai rumah memiliki non-linieritas dan interaksi yang lebih kompleks yang lebih baik ditangkap oleh algoritma berbasis pohon seperti Random Forest, serta kemampuannya yang lebih *robust* terhadap *multicollinearity* yang terdeteksi dalam data.

**Kesimpulan:**

Berdasarkan evaluasi menggunakan metrik MSE, RMSE, dan $R^2$, model **Random Forest Regressor** adalah pilihan yang lebih baik dan direkomendasikan untuk memprediksi nilai rumah median dalam proyek ini karena akurasinya yang lebih tinggi dan kemampuannya untuk menangani kompleksitas data. Model ini dapat menjadi alat yang efektif untuk membantu berbagai pihak di pasar properti dalam pengambilan keputusan berbasis data.

---
