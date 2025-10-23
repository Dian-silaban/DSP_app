import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
# Mengatur agar tampilan plot lebih baik
sns.set_theme(style="whitegrid")

df = pd.read_csv('employee_data.csv')
print("Shape of the dataframe:", df.shape)
print("\nInfo:")
df.info()


# 1. Menghapus kolom yang tidak relevan
df_cleaned = df.drop(['EmployeeId', 'EmployeeCount', 'StandardHours', 'Over18'], axis=1)

# 2. One-Hot Encoding untuk fitur kategorikal
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

print("Shape of dataframe after encoding:", df_encoded.shape)
print("\nContoh kolom baru hasil encoding:")
print([col for col in df_encoded.columns if 'Department_' in col])

# 3. Memisahkan data berdasarkan nilai Attrition
df_known = df_encoded[df_encoded['Attrition'].notna()].copy()
df_unknown = df_encoded[df_encoded['Attrition'].isna()].copy()

# Mengubah tipe data Attrition menjadi integer untuk training
df_known['Attrition'] = df_known['Attrition'].astype(int)

print(f"\nData yang diketahui Attrition-nya: {df_known.shape[0]} baris")
print(f"Data yang akan diprediksi: {df_unknown.shape[0]} baris")


plt.figure(figsize=(8, 5))
sns.countplot(x='Attrition', data=df_known)
plt.title('Distribusi Kelas Attrition')
plt.ylabel('Jumlah Karyawan')
plt.xticks([0, 1], ['Tidak Resign', 'Resign'])
plt.show()

print("Persentase Distribusi Kelas:")
print(df_known['Attrition'].value_counts(normalize=True) * 100)


# Memisahkan fitur (X) dan target (y)
X = df_known.drop('Attrition', axis=1)
y = df_known['Attrition']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inisialisasi dan melatih model
# Menggunakan class_weight='balanced' untuk menangani data imbalance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)

print("Model training selesai.")


# Melakukan prediksi pada data tes
y_pred = rf_classifier.predict(X_test)

# Menampilkan hasil evaluasi
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Tidak Resign', 'Resign']))

# Menampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Resign', 'Resign'], yticklabels=['Tidak Resign', 'Resign'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()


importances = rf_classifier.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("10 Fitur Teratas yang Mempengaruhi Attrition:")
print(feature_importance_df.head(10))

# Visualisasi
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('15 Fitur Paling Penting untuk Prediksi Attrition')
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()


# Mempersiapkan data yang akan diprediksi
X_unknown = df_unknown.drop('Attrition', axis=1)

# Memastikan urutan kolom sama persis dengan data training
X_unknown = X_unknown[X_train.columns]

# Melakukan prediksi
predictions_unknown = rf_classifier.predict(X_unknown)
print(f"Prediksi selesai untuk {len(predictions_unknown)} karyawan.")

# Mengisi nilai yang kosong pada dataframe asli dengan hasil prediksi
df_final = df_encoded.copy()
df_final.loc[df_final['Attrition'].isna(), 'Attrition'] = predictions_unknown

# Mengubah tipe data kembali menjadi integer
df_final['Attrition'] = df_final['Attrition'].astype(int)

print("\nContoh data setelah diisi:")
print(df_final[df['Attrition'].isna()].head())

# Menyimpan hasil ke file CSV baru
output_file = 'employee_data_predicted.csv'
df_final.to_csv(output_file, index=False)

print(f"\nHasil prediksi telah disimpan ke file '{output_file}'")


# --- Menyimpan Model dan Kolom ---
# Simpan model ke file
joblib.dump(rf_classifier, 'rf_model.joblib')
print("Model telah disimpan ke 'rf_model.joblib'")

# Simpan daftar kolom yang digunakan saat training
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')
print("Kolom model telah disimpan ke 'model_columns.pkl'")