# Laporan Proyek *machine learning* - Muhammad Bagus Adi Prayoga

## Daftar Isi

- [Laporan Proyek *machine learning* - Muhammad Bagus Adi Prayoga](#laporan-proyek-machine-learning---muhammad-bagus-adi-prayoga)
  - [Daftar Isi](#daftar-isi)
  - [Domain Proyek](#domain-proyek)
  - [Business Understanding](#business-understanding)
    - [Problem Statements](#problem-statements)
    - [Goals](#goals)
    - [Solution Statements](#solution-statements)
  - [Data Understanding](#data-understanding)
    - [*Descriptive Statistics*](#descriptive-statistics)
    - [*EDA: Uni-Variate Analysis*](#eda-uni-variate-analysis)
    - [*EDA: Multi-Variate Analysis*](#eda-multi-variate-analysis)
  - [*Data Preparation*](#data-preparation)
    - [*Outlier Handling*](#outlier-handling)
    - [*Splitting Data*](#splitting-data)
    - [*Feature Scaling (Standardization)*](#feature-scaling-standardization)
  - [*Modeling*](#modeling)
    - [*Decision Tree Classifier*](#decision-tree-classifier)
    - [*Support Vector Machine (SVM)*](#support-vector-machine-svm)
    - [*K-Nearest Neighbor*](#k-nearest-neighbor)
    - [*Voting Classifier*](#voting-classifier)
    - [Perbandingan Akurasi](#perbandingan-akurasi)
  - [Evaluation](#evaluation)
    - [*Cross Validation*](#cross-validation)
    - [*Confusion Matrix*](#confusion-matrix)
    - [*Classification Report*](#classification-report)
    - [Kesimpulan Evaluasi](#kesimpulan-evaluasi)
  - [Referensi](#referensi)


## Domain Proyek

Industri susu memainkan peran penting dalam menyediakan sumber nutrisi yang kaya bagi manusia. Susu sapi merupakan bahan baku utama dalam berbagai produk makanan dan minuman, dan kualitasnya sangat penting untuk memastikan keamanan dan gizi bagi konsumen. Standar kualitas yang ketat diterapkan dalam industri susu untuk memenuhi kebutuhan ini.

Klasifikasi kualitas susu sapi secara objektif dan konsisten merupakan langkah penting bagi perusahaan susu. Hal ini memberikan kepercayaan kepada konsumen bahwa produk yang mereka konsumsi aman dan berkualitas. Kualitas susu yang baik juga berdampak langsung pada nilai ekonominya di pasar, menghasilkan harga yang lebih tinggi dan meningkatkan daya jual [1]. Selain itu, kualitas susu mentah yang terjaga penting untuk menghasilkan produk susu akhir yang berkualitas tinggi [2].

Solusi yang diusulkan untuk meningkatkan klasifikasi kualitas susu sapi adalah dengan mengadopsi teknik analitik prediktif, khususnya menggunakan *machine learning*. Dengan memanfaatkan data tentang karakteristik susu, seperti pH, suhu, dan warna, model *machine learning* dapat menghasilkan penilaian kualitas yang lebih objektif dan konsisten. Pendekatan ini diharapkan dapat meningkatkan akurasi dalam penilaian kualitas susu sapi, serta membantu perusahaan dalam meningkatkan efisiensi proses evaluasi mereka.


## Business Understanding

### Problem Statements

Berdasarkan pemahaman atas domain proyek yang telah diuraikan sebelumnya, berikut adalah *problem statements* yang teridentifikasi:
- Fitur-fitur apa saja yang paling berpengaruh dalam menentukan kualitas susu sapi?
- Bagaimana langkah-langkah *pre-processing* data yang optimal dalam pengembangan model *machine learning* untuk melakukan klasifikasi kualitas susu sapi?
- Bagaimana merancang dan mengembangkan model *machine learning* yang mampu mengklasifikasikan kualitas susu sapi secara efektif?
- Bagaimana cara yang tepat untuk mengevaluasi kinerja model *machine learning* yang telah dibangun dalam mengklasifikasikan kualitas susu sapi?

### Goals

Berdasarkan *problem statements* yang telah diidentifikasi sebelumnya, berikut adalah beberapa *goals* dari proyek ini:
- Mengetahui fitur-fitur yang paling berpengaruh dalam menentukan kualitas susu sapi.
- Mengembangkan alur *pre-processing* data yang efisien dan efektif untuk mempersiapkan data masukan bagi model *machine learning* guna klasifikasi kualitas susu sapi.
- Membangun model *machine learning* yang mampu mengklasifikasikan kualitas susu sapi dengan tingkat akurasi minimal 90% dan *F1-score* minimal 90%.
- Melakukan evaluasi menyeluruh terhadap kinerja model *machine learning* yang telah dibangun untuk menentukan model terbaik yang memenuhi standar performa yang ditetapkan.

### Solution Statements

Berdasarkan *goals* di atas, maka diperoleh beberapa *solution statement* untuk menyelesaikan masalah tersebut, yaitu:
- Melakukan eksplorasi data untuk mengetahui fitur-fitur yang paling berpengaruh dalam menentukan kualitas susu sapi.
- Melakukan persiapan data untuk memastikan kesiapan dalam penggunaan dalam pembuatan model *machine learning*. Ini mencakup beberapa tahapan seperti *missing value handling*, *outliers handling*, *feature engineering*, *data splitting*, serta standarisasi.
- Mengembangkan model *machine learning* yang efektif dengan langkah-langkah berikut:
	- Membandingkan *baseline model* menggunakan *library* *LazyPredict* dan mengevaluasi kinerjanya.
	- Memilih tiga model *machine learning* dengan akurasi dan *F1-score* tertinggi dari hasil evaluasi serta mempertimbangkan kelebihan dan kekurangan model masing-masing.
	- Membangun model *Voting Classifier* yang menggabungkan tiga *baseline model* terbaik sebelumnya dengan tujuan untuk menggeneralisasi hasil prediksi yang lebih baik.
- Melakukan evaluasi terhadap setiap model menggunakan teknik *cross validation*, *confusion matrix*, serta berbagai *metrics performance* seperti akurasi, presisi, *recall*, dan *F1-Score* untuk menilai kinerja dan kemampuan prediktifnya secara menyeluruh.


## Data Understanding

Datasets yang digunakan pada kasus ini adalah datasets yang dikumpulkan secara manual melalui pengamatan yang bersumber dari [Kaggle - Milk Quality Prediction](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). Datasets ini berisi 1,059 baris dan 8 kolom, berikut adalah deskripsi dari masing-masing kolomnya:
- ***pH:*** Tingkat keasaman susu (ideal: 6.25 - 6.90).
- ***Temperature:*** Suhu susu (ideal: 34°C - 45.20°C).
- ***Taste:*** Rasa susu (0: Buruk, 1: Baik).
- ***Odor:*** Bau susu (0: Buruk, 1: Baik).
- ***Fat:*** Kandungan lemak dalam susu (0: Rendah, 1: Tinggi).
- ***Turbidity:*** Tingkat kekeruhan susu (0: Rendah, 1: Tinggi).
- ***Colour:*** Warna susu (kisaran: 240 - 255).
- ***Grade (Target Variable):*** Kualitas susu (*"Low"* - Buruk, *"Medium"* - Sedang, *"High"* - Baik).

### *Descriptive Statistics*

Tabel 1. Deskripsi Statistik Data

|       |         pH |   Temperature |       Taste |        Odor |         Fat |   Turbidity |     Colour |
|:------|-----------:|--------------:|------------:|------------:|------------:|------------:|-----------:|
| count | 1059       |     1059      | 1059        | 1059        | 1059        | 1059        | 1059       |
| mean  |    6.63012 |       44.2266 |    0.546742 |    0.432483 |    0.671388 |    0.491029 |  251.84    |
| std   |    1.39968 |       10.0984 |    0.498046 |    0.495655 |    0.46993  |    0.500156 |    4.30742 |
| min   |    3       |       34      |    0        |    0        |    0        |    0        |  240       |
| 25%   |    6.5     |       38      |    0        |    0        |    0        |    0        |  250       |
| 50%   |    6.7     |       41      |    1        |    0        |    1        |    0        |  255       |
| 75%   |    6.8     |       45      |    1        |    1        |    1        |    1        |  255       |
| max   |    9.5     |       90      |    1        |    1        |    1        |    1        |  255       |

Beberapa informasi yang dapat diambil dari *descriptive statistics* diatas adalah:
- Datasets ini terdiri dari 1,059 baris dan 8 kolom.
- Tidak ada *missing values* pada datasets ini.
- Ada indikasi *outlier* pada kolom ***pH*** dan ***Temperature***.
- Mayoritas sampel memiliki rasa "Baik", bau "Buruk", kandungan lemak "Tinggi", tingkat kekeruhan "Rendah", dan kualitas "Rendah".

### *EDA: Uni-Variate Analysis*

![Distribusi Label](assets/01_01.png)

Gambar 1. Distribusi Label
<br>

![Distribusi Fitur Kategori](assets/01_02.png)

Gambar 2. Distribusi Fitur Kategori
<br>

![Histogram Fitur Numerik](assets/01_03.png)

Gambar 3. Histogram Fitur Numerik
<br>

![Boxplot Fitur Numerik](assets/01_04.png)

Gambar 4. Boxplot Fitur Numerik
<br>

Informasi yang didapatkan dari *uni-variate analysis*:
- **Distribusi label yang tidak seimbang** dapat menghasilkan model yang memprediksi *"Low"* lebih sering, meskipun kualitas susu sebenarnya *"Medium"* atau *"High"*.
    - Salah satu solusi untuk hal ini jika dimungkinkan adalah dengan menggunakan teknik *sampling* seperti *SMOTE*.
    - Atau menggunakan model klasifikasi yang dapat bekerja pada distribusi data yang tidak seimbang seperti *Random Forest* atau *SVM*.
- Distribusi fitur kategorikal sudah cukup bagus, namun pada fitur ***Fat*** terdapar perbedaan jumlah yang sangat banyak.
- **Terdapat outlier pada semua fitur numerical**, sehingga perlu dilakukan **outlier handling**.

### *EDA: Multi-Variate Analysis*

Sebelum melakukan *multi-variate analysis*, dilakukan *label encoding* pada kolom ***Grade*** dengan menggunakan metode *ordinal encoding*. Alasan menggunakan *ordinal encoding* adalah karena kolom ***Grade*** merupakan kolom target yang memiliki tingkatan kualitas, yaitu:
- *"Low"* = 0
- *"Medium"* = 1
- *"High"* = 2

Selanjutnya, dilakukan visualisasi *heatmap* korelasi antar fitur. Hasil dari visualisasi *heatmap* korelasi menunjukkan:

![Heatmap Korelasi](assets/01_05.png)

Gambar 5. *Heatmap* Korelasi
<br>

Informasi yang didapatkan dari multivariate analysis:
- Kolom ***Grade*** memiliki **korelasi positif** yang kecil dengan kolom **Odor** dan ***Fat***.
- Kolom ***Grade*** memiliki **korelasi negatif** yang kecil dengan kolom ***Turbidity***.
- Kolom ***Grade*** memiliki **korelasi negatif** yang sedang dengan kolom ***Temperature***.


## *Data Preparation*

### *Outlier Handling*

Beberapa fitur dalam dataset, seperti **pH**, **Temperature**, dan **Colour**, memiliki *outlier*. *Outlier* adalah data yang nilainya jauh dari nilai mayoritas data. *Outlier* dapat menyebabkan model *machine learning* menjadi tidak akurat. Oleh karena itu, *outlier handling* dilakukan untuk menghilangkan *outlier* dari dataset. 

Metode yang digunakan untuk *outlier handling* adalah ***Interquartile Range (IQR)***. Metode *IQR (Interquartile Range)* digunakan untuk penanganan *outlier* karena metode ini menyediakan cara yang kuat dan efektif untuk mendeteksi dan menghapus *outlier* dari kumpulan data. Selain itu, metode ini juga mudah diinterpretasikan dan tidak memerlukan asumsi distribusi data [3]. 

Berikut adalah rumus dari metode IQR:
$$
IQR = Q3 - Q1 \\
Batas Bawah = Q1 - 1.5 \times IQR \\
Batas Atas = Q3 + 1.5 \times IQR \\
Outlier = Data < Batas Bawah \text{ or } Data > Batas Atas
$$

Hasil dari *outlier handling* menunjukkan:
- Data sebelum *outlier handling*: 1.059 baris
- Data setelah *outlier handling*: 648 baris
- Total *outlier* yang dihilangkan: 411 baris
- Rincian *outlier* yang dihilangkan:
  - pH: 379 baris
  - Temperatur: 103 baris
  - Warna: 32 baris

Berikut adalah visualisasi data hasil setelah *outlier handling*:
![Boxplot Fitur Numerik Setelah *Outlier Handling*](assets/02_01.png)

Gambar 6. Boxplot Fitur Numerik Setelah *Outlier Handling*
<br>

![Histogram Fitur Numerik Setelah *Outlier Handling*](assets/02_02.png)

Gambar 7. Histogram Fitur Numerik Setelah *Outlier Handling*
<br>

![Heatmap Korelasi Setelah *Outlier Handling*](assets/02_03.png)

Gambar 8. *Heatmap* Korelasi Setelah *Outlier Handling*
<br>

Informasi yang didapatkan dari hasil *outlier handling*:
- Setelah dilakukan *outlier handling*, distribusi data menjadi lebih baik dan tidak terdapat *outlier* pada fitur numerik.
- Korelasi antar fitur mengalami perubahan yang signifikan setelah dilakukan *outlier handling*.

### *Splitting Data*

Data yang sudah bersih kemudian dipecah menjadi dua bagian, yaitu *training set* dan *testing set*. Berikut adalah rincian dari *splitting data*:
- **Training Set**: 80% dari total data
- **Testing Set**: 20% dari total data

Tujuan dari splitting data adalah untuk mempersiapkan data yang akan digunakan untuk melatih model *machine learning* dan data yang akan digunakan untuk menguji model *machine learning*.

### *Feature Scaling (Standardization)*

*Feature scaling* dilakukan untuk menstandarisasi nilai-nilai dari fitur-fitur numerik dalam dataset. Hal ini dilakukan untuk memastikan bahwa semua fitur memiliki skala yang sama. Metode yang digunakan adalah ***StandardScaler***.

***StandardScaler*** mengubah distribusi data sehingga memiliki rata-rata 0 dan standar deviasi 1. Hal ini dilakukan dengan mengurangi setiap nilai dengan rata-rata dan kemudian membaginya dengan standar deviasi.

## *Modeling*

Proses pemodelan dibagi menjadi beberapa tahapan, yaitu:
- ***Baseline Model***: Melakukan pemodelan menggunakan beberapa algoritma *machine learning* dengan bantuan *library LazyPredict*.
- ***Model Selection***: Memilih tiga model *machine learning* dengan akurasi dan *F1-score* tertinggi dari hasil evaluasi.
- ***Model Development***: Membangun model *Voting Classifier* yang menggabungkan tiga *baseline model* terbaik sebelumnya.

*Baseline model* dibangun menggunakan *library LazyPredict*. Library ini digunakan untuk membandingkan performa dari berbagai algoritma *machine learning* dengan cepat. *Baseline model* yang akan dipilih adalah **model traditional *machine learning***. Berikut adalah hasil dari perbandingan dengan menggunakan *library LazyPredict*:

Tabel 2. Hasil Evaluasi Baseline Model
| Model                         |   Accuracy |   Balanced Accuracy | ROC AUC   |   F1 Score |   Time Taken |
|:------------------------------|-----------:|--------------------:|:----------|-----------:|-------------:|
| LGBMClassifier                |   1        |            1        |           |   1        |    0.232165  |
| XGBClassifier                 |   1        |            1        |           |   1        |    0.073621  |
| DecisionTreeClassifier        |   1        |            1        |           |   1        |    0.0174434 |
| ExtraTreesClassifier          |   1        |            1        |           |   1        |    0.136188  |
| RandomForestClassifier        |   1        |            1        |           |   1        |    0.181055  |
| LabelPropagation              |   1        |            1        |           |   1        |    0.0340152 |
| LabelSpreading                |   1        |            1        |           |   1        |    0.0485413 |
| BaggingClassifier             |   1        |            1        |           |   1        |    0.0483181 |
| ExtraTreeClassifier           |   0.992308 |            0.994949 |           |   0.992314 |    0.0152066 |
| QuadraticDiscriminantAnalysis |   0.992308 |            0.994949 |           |   0.992314 |    0.0153189 |
| SVC                           |   0.992308 |            0.966667 |           |   0.992141 |    0.0195484 |
| KNeighborsClassifier          |   0.992308 |            0.966667 |           |   0.992141 |    0.0258548 |
| LogisticRegression            |   0.984615 |            0.960494 |           |   0.984492 |    0.0539668 |
| SGDClassifier                 |   0.984615 |            0.960494 |           |   0.984492 |    0.0190806 |
| LinearSVC                     |   0.984615 |            0.960494 |           |   0.984492 |    0.0443156 |
| CalibratedClassifierCV        |   0.984615 |            0.960494 |           |   0.984492 |    0.104081  |
| PassiveAggressiveClassifier   |   0.976923 |            0.956566 |           |   0.976772 |    0.0179486 |
| Perceptron                    |   0.969231 |            0.92211  |           |   0.968606 |    0.0222425 |
| AdaBoostClassifier            |   0.876923 |            0.905724 |           |   0.875516 |    0.115924  |
| LinearDiscriminantAnalysis    |   0.861538 |            0.861728 |           |   0.856756 |    0.0239906 |
| NearestCentroid               |   0.838462 |            0.852189 |           |   0.84233  |    0.0230031 |
| GaussianNB                    |   0.884615 |            0.807744 |           |   0.887382 |    0.0198004 |
| RidgeClassifier               |   0.869231 |            0.787542 |           |   0.865791 |    0.0209899 |
| RidgeClassifierCV             |   0.861538 |            0.754209 |           |   0.856306 |    0.0170121 |
| BernoulliNB                   |   0.784615 |            0.590572 |           |   0.776403 |    0.0185149 |
| DummyClassifier               |   0.507692 |            0.333333 |           |   0.341915 |    0.0158775 |

Selanjutnya, *model selection* dilakukan dengan memilih tiga *traditional machine learning* model dengan akurasi dan *F1-score* tertinggi.


### *Decision Tree Classifier*

*Decision Tree Classifier* merupakan model yang memiliki struktur pohon keputusan mudah dipahami, memungkinkan pelacakan pengaruh fitur terhadap prediksi akhir. Model ini cepat dilatih dan tidak memerlukan scaling data. Kelemahannya, *Decision Tree Classifier* rentan terhadap *overfitting*, memiliki variansi tinggi, dan bias terhadap fitur kategoris dengan banyak level [4].

### *Support Vector Machine (SVM)*

*Support Vector Machine (SVM)* merupakan model yang efektif untuk data dengan dimensi tinggi dan mampu menangani hubungan *non-linear* antara fitur dan label. Model ini mencegah *overfitting* dengan *margin classification*, mencari *hyperplane* dengan margin terbesar untuk memisahkan kelas. Kelemahan *SVM* adalah waktu pelatihan yang lama untuk dataset besar dan sensitivitas terhadap *noise* dalam data [5].

### *K-Nearest Neighbor*

*K-Nearest Neighbor (KNN)* merupakan model sederhana dan mudah diimplementasikan yang tidak membuat asumsi tentang distribusi data, sehingga cocok untuk data non-normal. Kelebihan *KNN* lainnya adalah tidak memerlukan waktu pelatihan dan dapat langsung digunakan. Kelemahan *KNN* adalah kinerjanya yang menurun pada data dimensi tinggi (*curse of dimensionality*), sensitivitas terhadap data imbang (*imbalanced data*), dan *outlier* yang dapat mempengaruhi prediksi secara signifikan [6].

### *Voting Classifier*

Dengan mempertimbangkan kelebihan dan kekurangan dari masing-masing model, maka akan dikembangkan model ***Voting Classifier*** yang menggabungkan tiga model *machine learning* terbaik sebelumnya.  

*Voting Classifier* menawarkan solusi untuk meningkatkan akurasi dan meminimalisasi *overfitting* melalui kombinasi hasil dari beberapa model berbeda [6]. Kelebihannya terletak pada kemudahan implementasi dan pemahaman. Namun, kompleksitas dan waktu training yang lebih tinggi dibandingkan model tunggal menjadi pertimbangan penting. Selain itu, *Voting Classifier* memiliki tingkat interpretabilitas yang lebih rendah karena menggabungkan beberapa model.

### Perbandingan Akurasi

Berikut adalah perbandingan performa dari tiga model *machine learning* terbaik dan model *Voting Classifier*:

Tabel 3. Perbandingan Performa Model
|              |   Score Train |   Score Test |
|:-------------|--------------:|-------------:|
| SVC          |      0.996139 |     0.992308 |
| DecisionTree |      1        |     1        |
| KNN          |      0.994208 |     0.992308 |
| Voting       |      0.998069 |     0.992308 |

Dari hasil perbandingan performa di atas, dapat disimpulkan bahwa model ***Decision Tree*** memiliki performa terbaik dengan akurasi 100%, baik pada data *train* maupun data *test*. Akan tetapi, model ini memiliki kecenderungan untuk *overfitting*. Oleh karena itu, perlu dilakukan evaluasi lebih lanjut pada semua model yang telah dibangun.

## Evaluation

### *Cross Validation*

***Cross-validation (CV)*** adalah metode statistik yang dapat digunakan untuk mengevaluasi kinerja model atau algoritma dimana data dipisahkan menjadi dua subset yaitu *training data* dan *validation data*. Model atau algoritma dilatih oleh *subset training* dan divalidasi oleh *subset validation*.

*Cross validation* digunakan untuk menghindari *overfitting* dan *underfitting* pada model. *Cross validation* juga dapat digunakan untuk memilih model yang paling baik untuk digunakan pada dataset yang diberikan.

Tabel 4. Hasil *Cross Validation*
|              |   Score Train |   Score Test |   CV Mean |    CV Std |
|:-------------|--------------:|-------------:|----------:|----------:|
| SVC          |      0.996139 |     0.992308 |  0.990309 | 0.0129574 |
| DecisionTree |      1        |     1        |  0.994193 | 0.0123554 |
| KNN          |      0.994208 |     0.992308 |  0.986501 | 0.0150346 |
| Voting       |      0.998069 |     0.992308 |  0.992232 | 0.0128252 |

Hasil *cross validation* menunjukkan performa ***Decision Tree*** yang sangat baik. Model ini mencapai skor sempurna pada data *training* dan *testing*, serta nilai rata-rata *cross validation* yang tinggi. Di sisi lain, model *Voting Classifier* juga menunjukkan performa kuat dan stabil pada semua metrik evaluasi.

### *Confusion Matrix*

***Confusion matrix*** adalah tabel yang digunakan untuk mengevaluasi kinerja dari suatu model *machine learning* terutama pada task klasifikasi. *Confusion matrix* menunjukkan jumlah prediksi yang benar dan yang salah yang dibagi berdasarkan kelas target.

![Confusion Matrix](assets/03_01.jpg)

Gambar 9. Penjelasan *Confusion Matrix*
<br>

Keterangan:
- ***True Positive (TP)***: Prediksi positif yang benar
- ***True Negative (TN)***: Prediksi negatif yang benar
- ***False Positive (FP)***: Prediksi positif yang salah
- ***False Negative (FN)***: Prediksi negatif yang salah

Beriikut adalah hasil dari *confusion matrix* untuk model yang telah dibangun:
```
Confusion Matrix SVC:
 [[ 9  0  1]
 [ 0 66  0]
 [ 0  0 54]]

Confusion Matrix Decision Tree:
 [[10  0  0]
 [ 0 66  0]
 [ 0  0 54]]

Confusion Matrix KNN:
 [[ 9  0  1]
 [ 0 66  0]
 [ 0  0 54]]

Confusion Matrix Voting:
 [[ 9  0  1]
 [ 0 66  0]
 [ 0  0 54]]
```

Secara umum, dari hasil *confusion matrix*, ***Decision Tree*** adalah model yang paling baik dalam melakukan klasifikasi. Model ***Decision Tree*** berhasil mengklasifikasikan semua sampel dengan benar, tanpa ada kesalahan klasifikasi. Sedangkan model *SVC*, *KNN*, dan *Voting memiliki* kinerja yang serupa dengan kebanyakan sampel diklasifikasikan dengan benar, namun dengan beberapa kesalahan yang terjadi.

### *Classification Report*

***Classification report*** adalah laporan yang digunakan untuk mengevaluasi kinerja dari suatu model *machine learning* terutama pada task klasifikasi. *Classification report* menunjukkan beberapa *metrics performance* seperti *accuracy*, *precision*, *recall*, dan *F1-score*. Berikut adalah penjelasan dari masing-masing *metrics performance*:
1. ***Accuracy***: Rasio prediksi benar dari keseluruhan prediksi yang dilakukan oleh model.
2. ***Precision***: Rasio prediksi benar positif dibandingkan dengan keseluruhan prediksi positif yang dilakukan oleh model.
3. ***Recall***: Rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
4. ***F1-Score***: Rata-rata harmonik dari *precision* dan *recall*.

![Rumus Metrics Performance](assets/03_02.webp)

Gambar 10. Rumus *Metrics Performance*
<br>

Beriikut adalah hasil dari *classification report* untuk model yang telah dibangun:
```
Classification report SVC:
              precision    recall  f1-score   support

         0.0       1.00      0.90      0.95        10
         1.0       1.00      1.00      1.00        66
         2.0       0.98      1.00      0.99        54

    accuracy                           0.99       130
   macro avg       0.99      0.97      0.98       130
weighted avg       0.99      0.99      0.99       130

---

Classification report Decision Tree:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00        10
         1.0       1.00      1.00      1.00        66
         2.0       1.00      1.00      1.00        54

    accuracy                           1.00       130
   macro avg       1.00      1.00      1.00       130
weighted avg       1.00      1.00      1.00       130

---

Classification report KNN:
              precision    recall  f1-score   support

         0.0       1.00      0.90      0.95        10
         1.0       1.00      1.00      1.00        66
         2.0       0.98      1.00      0.99        54

    accuracy                           0.99       130
   macro avg       0.99      0.97      0.98       130
weighted avg       0.99      0.99      0.99       130

---

Classification report Voting Classifier:
              precision    recall  f1-score   support

         0.0       1.00      0.90      0.95        10
         1.0       1.00      1.00      1.00        66
         2.0       0.98      1.00      0.99        54

    accuracy                           0.99       130
   macro avg       0.99      0.97      0.98       130
weighted avg       0.99      0.99      0.99       130
```

Dari interpretasi tersebut, dapat disimpulkan bahwa semua model memiliki performa yang sangat baik dengan akurasi yang tinggi dan *f1-score* yang baik pula. Namun, model ***Decision Tree*** memperoleh hasil yang sempurna dengan akurasi dan *f1-score* 1.00 untuk setiap kelas, sehingga bisa dianggap sebagai model terbaik dalam hal ini.

### Kesimpulan Evaluasi

Berdasarkan hasil evaluasi, tidak terdapat perbedaan signifikan antara performa model pada data *train*, *test*, dan *cross validation*. Hal ini menunjukkan bahwa model yang dibangun memiliki hasil yang cukup baik dan kecil kemungkinan mengalami *overfitting* atau *underfitting*.

Dari hasil evaluasi, ***Decision Tree*** menunjukkan **performa terbaik** dengan akurasi dan *F1-score* sempurna pada setiap kelas. Hal ini menjadikannya model ideal untuk klasifikasi kualitas susu. Namun, perlu diingat bahwa ***Decision Tree*** memiliki kecenderungan untuk ***overfitting***. Oleh karena itu, ***Voting Classifier*** juga menjadi pilihan yang patut dipertimbangkan.

***Voting Classifier*** merupakan model ensemble yang menggabungkan tiga model *machine learning* terbaik (*SVC*, *Decision Tree*, dan *KNN*). Model ini memiliki performa kuat dan stabil pada semua metrik evaluasi, serta memiliki kelebihan dalam mengurangi overfitting.

## Referensi

*[1] G. Castellini, S. Barello, dan A. C. Bosio, “Milk Quality Conceptualization: A Systematic Review of Consumers’, Farmers’, and Processing Experts’ Views,” Foods, vol. 12, no. 17, hlm. 3215, Agu 2023, doi: 10.3390/foods12173215.*

*[2] A. Brodziak, J. Wajs, M. Zuba-Ciszewska, J. Król, M. Stobiecka, dan A. Jańczuk, “Organic versus Conventional Raw Cow Milk as Material for Processing,” Animals, vol. 11, no. 10, hlm. 2760, Sep 2021, doi: 10.3390/ani11102760.*

*[3] H.-M. Kaltenbach, A Concise Guide to Statistics. dalam SpringerBriefs in Statistics. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012. doi: 10.1007/978-3-642-23502-3.*

*[4] V. Y. Kulkarni, P. K. Sinha, dan M. C. Petare, “Weighted Hybrid Decision Tree Model for Random Forest Classifier,” J. Inst. Eng. India Ser. B, vol. 97, no. 2, hlm. 209–217, Jun 2016, doi: 10.1007/s40031-014-0176-y.*

*[5] D. J. Kalita, V. P. Singh, dan V. Kumar, “A Survey on SVM Hyper-Parameters Optimization Techniques,” dalam Social Networking and Computational Intelligence, vol. 100, R. K. Shukla, J. Agrawal, S. Sharma, N. S. Chaudhari, dan K. K. Shukla, Ed., dalam Lecture Notes in Networks and Systems, vol. 100. , Singapore: Springer Singapore, 2020, hlm. 243–256. doi: 10.1007/978-981-15-2071-6_20.*

*[6] K. Taunk, S. De, S. Verma, dan A. Swetapadma, “A Brief Review of Nearest Neighbor Algorithm for Learning and Classification,” dalam 2019 International Conference on Intelligent Computing and Control Systems (ICCS), Madurai, India: IEEE, Mei 2019, hlm. 1255–1260. doi: 10.1109/ICCS45141.2019.9065747.*

*[7] I. Gandhi dan M. Pandey, “Hybrid Ensemble of classifiers using voting,” dalam 2015 International Conference on Green Computing and Internet of Things (ICGCIoT), Greater Noida, Delhi, India: IEEE, Okt 2015, hlm. 399–404. doi: 10.1109/ICGCIoT.2015.7380496.*