# %% [markdown]
# # Laporan Proyek Machine Learning - Muhammad Bagus Adi Prayoga

# %% [markdown]
# ## Problem Domain

# %% [markdown]
# Susu sapi adalah komponen penting dalam rantai pangan manusia. Memastikan kualitas susu yang tinggi adalah suatu keharusan untuk memenuhi standar keamanan dan gizi yang diharapkan oleh konsumen. Standar kualitas yang ketat diterapkan dalam industri susu untuk memastikan produk yang dihasilkan memenuhi persyaratan yang ditetapkan.
# 
# Namun, kualitas susu dapat bervariasi dipengaruhi oleh sejumlah faktor, termasuk kesehatan sapi, kondisi lingkungan, dan proses produksi. Variabilitas ini dapat menyebabkan fluktuasi dalam kualitas susu, dan dalam beberapa kasus, produk mungkin tidak memenuhi standar yang diinginkan.
# 
# Pengklasifikasian kualitas susu sapi saat ini umumnya masih bergantung pada metode organoleptik (penilaian melalui indra manusia seperti bau, rasa, dan warna). Metode ini cenderung subjektif dan tidak konsisten, menyebabkan penilaian kualitas yang kurang akurat.
# 
# Penggunaan teknik analitik prediktif, seperti Machine Learning, dapat menjadi solusi untuk mengklasifikasikan kualitas susu sapi secara lebih objektif dan konsisten. Model Machine Learning dapat mengevaluasi parameter kimia susu, seperti pH, lemak, protein, dan laktosa, untuk menentukan kualitasnya. Pendekatan ini diharapkan dapat meningkatkan konsistensi dan akurasi dalam penilaian kualitas susu sapi, serta membantu dalam meningkatkan efisiensi proses evaluasi.

# %% [markdown]
# ## Business Understanding

# %% [markdown]
# ### Problem Statements
# 
# Berdasarkan pemahaman atas domain proyek yang telah diuraikan sebelumnya, berikut adalah problem statements yang teridentifikasi:
# - Bagaimana langkah-langkah pre-processing data yang optimal dalam pengembangan model Machine Learning untuk melakukan klasifikasi kualitas susu sapi?
# - Bagaimana kita dapat merancang dan mengembangkan model Machine Learning yang mampu mengklasifikasikan kualitas susu sapi secara efektif?
# - Bagaimana cara yang tepat untuk mengevaluasi kinerja model Machine Learning yang telah dibangun dalam mengklasifikasikan kualitas susu sapi?
# 
# ### Goals
# 
# Berdasarkan problem statements yang telah diidentifikasi sebelumnya, berikut adalah beberapa goals dari proyek ini:
# - Mengembangkan alur pre-processing data yang efisien dan efektif untuk mempersiapkan data masukan bagi model Machine Learning guna klasifikasi kualitas susu sapi.
# - Membangun model Machine Learning yang mampu mengklasifikasikan kualitas susu sapi dengan tingkat akurasi minimal 90% dan F1-score minimal 0.9.
# - Melakukan evaluasi menyeluruh terhadap kinerja model Machine Learning yang telah dibangun untuk menentukan model terbaik yang memenuhi standar performa yang ditetapkan.
# 
# ### Solution Statements
# 
# Berdasarkan goals di atas, maka diperoleh beberapa solution statement untuk menyelesaikan masalah tersebut, yaitu:
# - Melakukan persiapan data untuk memastikan kesiapan dalam penggunaan dalam pembuatan model machine learning. Ini mencakup beberapa tahapan seperti penanganan missing value, outliers, feature engineering, pemisahan data, serta standarisasi.
# - Mengembangkan model machine learning yang efektif dengan langkah-langkah berikut:
# 	- Membandingkan baseline model menggunakan library LazyPredict dan mengevaluasi kinerjanya.
# 	- Memilih tiga model machine learning dengan akurasi dan F1-score tertinggi dari hasil evaluasi.
# 	- Membangun model Voting Classifier yang menggabungkan tiga baseline model terbaik sebelumnya.
# - Melakukan evaluasi terhadap setiap model menggunakan teknik cross validation, confusion matrix, serta berbagai metrics performance seperti akurasi, presisi, recall, dan F1-Score untuk menilai kinerja dan kemampuan prediktifnya secara menyeluruh.

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# ### Initial Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
! pip -q install kaggle
! mkdir ~/.kaggle
! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download cpluzshrijayan/milkquality
! unzip milkquality.zip

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("milknew.csv")
df.head()

# %% [markdown]
# #### Exploratory Data Analysis: Descriptive Statistics

# %%
# melihat bentuk data
df.shape

# %%
# informasi kolom dan datatype
df.columns, df.info()

# %%
# mengubah kolom ini menjadi kategorikal (karena hanya mengandung nilai 0 atau 1 saja)
df.rename(columns={'Fat ': 'Fat'}, inplace=True)
df.rename(columns={'Temprature': 'Temperature'}, inplace=True)

# for col in ["Taste", "Odor", "Fat", "Turbidity", "Grade"]:
for col in ["Grade"]:
    df[col] = df[col].astype("category")

df.info()

# %%
# mengecek null value
df.isna().sum()

# %%
# statistik deskriptif
df.describe()

# %%
df.describe(include=["category"])

# %% [markdown]
# Beberapa informasi yang dapat diambil dari descriptive statistics diatas adalah:
# - Datasets ini terdiri dari 1,059 baris dan 8 kolom
# - Tidak ada missing values pada datasets ini
# - Ada indikasi outlier pada kolom pH dan Temperature
# - Mayoritas sampel memiliki rasa "Baik", bau "Buruk", kandungan lemak "Tinggi", tingkat kekeruhan "Rendah", dan kualitas "Rendah".

# %% [markdown]
# #### Exploratory Data Analysis: Univariate Analysis

# %%
# melihat persentase label
count = df["Grade"].value_counts()
percentage = df["Grade"].value_counts(normalize=True) * 100
df_label = pd.DataFrame({'count': count, 'percentage': percentage})
print(df_label)

# countplot untuk melihat distribusi label
plt.figure(figsize=(6, 4))
sns.set_style('whitegrid')
sns.countplot(x="Grade", data=df, order=["low", "medium", "high"])
plt.bar_label(plt.gca().containers[0])
plt.title('Distribusi Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# %%
# countplot untuk melihat distribusi data fitur kategorikal
categorical_columns = ["Taste", "Odor", "Fat", "Turbidity"]

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(6, 5))
for variable, subplot in zip(categorical_columns, ax.flatten()):
    subplot.set_title(f'Distribusi {variable}')
    sns.countplot(x=variable, data=df, ax=subplot)

plt.tight_layout()
plt.show()

# %%
# histogram untuk melihat distribusi data fitur numerical
numerical_columns = ["pH", "Temperature", "Colour"]

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for variable, subplot in zip(numerical_columns, ax.flatten()):
    subplot.set_title(f'Histogram {variable}')
    sns.histplot(x=variable, data=df, ax=subplot, kde=True)

plt.tight_layout()
plt.show()

# %%
# boxlot untuk melihat outlier data fitur numerical
numerical_columns = ["pH", "Temperature", "Colour"]

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for variable, subplot in zip(numerical_columns, ax.flatten()):
    subplot.set_title(f'Boxplot {variable}')
    sns.boxplot(x=variable, data=df, ax=subplot)

plt.tight_layout()
plt.show()

# %% [markdown]
# Informasi yang didapatkan dari univariate analysis:
# - Distribusi label yang tidak seimbang dapat menghasilkan model yang memprediksi "Low" lebih sering, meskipun kualitas susu sebenarnya "Medium" atau "High".
#     - Salah satu solusi untuk hal ini jika dimungkinkan adalah dengan menggunakan teknik sampling seperti SMOTE
#     - Menggunakan model klasifikasi yang dapat bekerja pada distribusi data yang tidak seimbang seperti Random Forest atau SVM
# - Distribusi fitur kategorikal sudah cukup bagus, namun pada fitur Fat terdapar perbedaan jumlah yang sangat banyak
# - Terdapat outlier pada semua fitur numerical, sehingga perlu dilakukan outlier handling

# %% [markdown]
# #### Exploratory Data Analysis: Multivariate Analysis

# %%
# encoding target label / Grade
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
df[["Grade"]] = encoder.fit_transform(df[["Grade"]])
print(encoder.categories_)
df.head()

# %%
# melihat pairplot untuk melihat hubungan antar fitur
sns.pairplot(df, palette='Set1', diag_kind='kde', hue="Grade")
plt.gcf().set_size_inches(16, 16)
plt.show()

# %%
# melihat korelasi antar kolom
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# %% [markdown]
# Informasi yang didapatkan dari multivariate analysis:
# - Kolom Grade memiliki korelasi positif yang kecil dengan kolom Odor dan Fat
# - Kolom Grade memiliki korelasi negatif yang kecil dengan kolom Turbidity
# - Kolom Grade memiliki korelasi negatif yang sedang dengan kolom Temperature

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Handling Outlier

# %%
# handling outlier menggunakan metode IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = df[(df < lower_fence) | (df > upper_fence)].count()

# melihat outlier
print("Total Data:", df.shape[0])
print("Total Outliers:", outliers.sum())
outliers

# %%
# exclude outlier
df = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

# %%
# melihat bentuk data setelah handling outlier
df.shape

# %%
# boxlot fitur setelah outlier handling

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for variable, subplot in zip(numerical_columns, ax.flatten()):
    subplot.set_title(f'Boxplot {variable}')
    sns.boxplot(x=variable, data=df, ax=subplot)

plt.tight_layout()
plt.show()

# %%
# distribusi data setelah outlier handling

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for variable, subplot in zip(numerical_columns, ax.flatten()):
    subplot.set_title(f'Histogram {variable}')
    sns.histplot(x=variable, data=df, ax=subplot, kde=True)

plt.tight_layout()
plt.show()

# %%
# melihat korelasi antar kolom setelah outlier handling
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# %% [markdown]
# ### Splitting Data

# %%
# memisahkan data menjadi fitur dan label
X = df.drop('Grade', axis=1)
y = df['Grade']

# %%
# Splitting the dataset into the Training set and Test set
import sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %% [markdown]
# ### Standardization

# %%
# standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

col = X_train.columns

X_train[col] = sc.fit_transform(X_train[col])
X_test[col] = sc.transform(X_test[col])

# %%
# melihat data setelah standardize X_train
X_train.head()

# %%
# melihat data setelah standardize X_test
X_test.head()

# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Train Baseline Model

# %%
!pip -q install lazypredict

import lazypredict
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models

# %% [markdown]
# Dari perbandingan performa LazyPredict diatas, maka dipilih tiga traditional ML model yaitu SVC, DecisionTreeClassifier, dan KNeighborsClassifier.

# %%
temp = models.to_markdown()
print(temp)

# %% [markdown]
# ### Support Vector Classification (SVC)

# %%
from sklearn.svm import SVC

svc_model = SVC()
svc_model = svc_model.fit(X_train, y_train)

# prediksi y
y_pred_svc_model = svc_model.predict(X_test)

# melihat hasil akurasinya
score_train_svc_model = svc_model.score(X_train, y_train)
score_test_svc_model = svc_model.score(X_test, y_test)
print("Score Train:", score_train_svc_model)
print("Score Test:", score_test_svc_model)

# %% [markdown]
# ### Decision Tree Classifier

# %%
from sklearn.tree import DecisionTreeClassifier

dtc_model = DecisionTreeClassifier()
dtc_model = dtc_model.fit(X_train, y_train)

# prediksi y
y_pred_dtc_model = dtc_model.predict(X_test)

# melihat hasil akurasinya
score_train_dtc_model = dtc_model.score(X_train, y_train)
score_test_dtc_model = dtc_model.score(X_test, y_test)
print("Score Train:", score_train_dtc_model)
print("Score Test:", score_test_dtc_model)

# %% [markdown]
# ### KNN

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model = knn_model.fit(X_train, y_train)

# prediksi y
y_pred_knn_model = knn_model.predict(X_test)

# melihat hasil akurasinya
score_train_knn_model = knn_model.score(X_train, y_train)
score_test_knn_model = knn_model.score(X_test, y_test)
print("Score Train:", score_train_knn_model)
print("Score Test:", score_test_knn_model)

# %% [markdown]
# ### Ensemble Model (Voting Classifier)

# %%
from sklearn.ensemble import VotingClassifier
from sklearn import metrics

# membuat base model voting
clf_list = [('SVC', svc_model), ('DecisionTreeClassifier', dtc_model), ('KNN', knn_model)]

# Melihat masing-masing akurasi dari berbagai model
for model_tuple in clf_list:
    model_temp = model_tuple[1]
    model_temp.fit(X_train, y_train)
    y_pred_temp = model_temp.predict(X_test)
    acc = metrics.accuracy_score(y_pred_temp, y_test)
    print(f"{model_tuple[0]}'s accuraccy : {acc:.2f}")

# %%
voting_model = VotingClassifier(clf_list, voting='hard')
voting_model.fit(X_train, y_train)

y_pred_voting_model = voting_model.predict(X_test)

# melihat hasil akurasinya
score_train_voting_model = voting_model.score(X_train, y_train)
score_test_voting_model = voting_model.score(X_test, y_test)

print("Score Train:", score_train_voting_model)
print("Score Test:", score_test_voting_model)

# %% [markdown]
# Dari hasil ketiga model di atas, dapat dikatakan sudah sangat bagus karena perbedaan antara akurasi train dan test tidak berbeda jauh. Akan tetapi, masih ada indikasi overfit karena terlalu sempurna sehingga akan dicoba evaluasi cross validation nanti.

# %% [markdown]
# ## Evaluation

# %%
# melakukan cross validation untuk mengecek overfitting
from sklearn.model_selection import cross_val_score

model_list = [('SVC', svc_model), ('DecisionTreeClassifier', dtc_model), ('KNN', knn_model), ('VotingClassifier', voting_model)]
val_result = []

for model in model_list:
    scores = cross_val_score(model[1], X_train, y_train, cv=10, scoring='accuracy')
    val_result.append(scores)
    print(f"{model[0]}: {scores.mean():.2f} (+/- {scores.std():.2f})")

# %%
# perbandingan akurasi dari berbagai model
acc = pd.DataFrame(
    {
        'Score Train': [score_train_svc_model, score_train_dtc_model, score_train_knn_model, score_train_voting_model],
        'Score Test': [score_test_svc_model, score_test_dtc_model, score_test_knn_model, score_test_voting_model]
    },
    index=['SVC', 'DecisionTree', 'KNN', 'Voting'])
acc['CV Mean'] = [val_result[0].mean(), val_result[1].mean(), val_result[2].mean(), val_result[3].mean()]
acc['CV Std'] = [val_result[0].std(), val_result[1].std(), val_result[2].std(), val_result[3].std()]
acc

# %%
print(acc.to_markdown())

# %%
# melihat confusion matrix
cm_svc = metrics.confusion_matrix(y_test, y_pred_svc_model)
cm_dtc = metrics.confusion_matrix(y_test, y_pred_dtc_model)
cm_knn = metrics.confusion_matrix(y_test, y_pred_knn_model)
cm_voting = metrics.confusion_matrix(y_test, y_pred_voting_model)

print("Confusion Matrix SVC:\n", cm_svc)
print("\nConfusion Matrix Decision Tree:\n", cm_dtc)
print("\nConfusion Matrix KNN:\n", cm_knn)
print("\nConfusion Matrix Voting:\n", cm_voting)

# %%
# melihat classification report
cr_svc = metrics.classification_report(y_test, y_pred_svc_model)
cr_dtc = metrics.classification_report(y_test, y_pred_dtc_model)
cr_knn = metrics.classification_report(y_test, y_pred_knn_model)
cr_voting = metrics.classification_report(y_test, y_pred_voting_model)

# %%
print("Classification report SVC:")
print(cr_svc)

# %%
print("Classification report Decision Tree:")
print(cr_dtc)

# %%
print("Classification report KNN:")
print(cr_knn)

# %%
print("Classification report Voting Classifier:")
print(cr_voting)


