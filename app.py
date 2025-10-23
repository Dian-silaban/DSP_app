from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Mengizinkan akses dari browser (frontend)

# === Muat model dan kolom ===
try:
    model = joblib.load('rf_model.joblib')
    model_columns = joblib.load('model_columns.pkl')
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file 'rf_model.joblib' dan 'model_columns.pkl' ada.")
    print("Jalankan 'train_and_save_model.py' terlebih dahulu.")
    model = None
    model_columns = None


# === ROUTE UNTUK HALAMAN UTAMA ===
@app.route('/')
def home():
    return render_template('home.html')  # arahkan ke templates/home.html


# === ROUTE UNTUK PREDIKSI (API) ===
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not model_columns:
        return jsonify({'error': 'Model tidak berhasil dimuat di server.'}), 500

    # Ambil data dari JSON body
    json_data = request.get_json()
    query_df = pd.DataFrame([json_data])

    # Contoh preprocessing sesuai model
    if 'OverTime' in query_df.columns:
        query_df['OverTime_Yes'] = query_df['OverTime']
        query_df = query_df.drop('OverTime', axis=1)

    # Reindex agar kolom sama dengan model
    query_final = pd.DataFrame(columns=model_columns)
    query_final = pd.concat([query_final, query_df], ignore_index=True).fillna(0)
    query_final = query_final[model_columns]

    # Prediksi probabilitas
    prediction_proba = model.predict_proba(query_final)
    attrition_probability = prediction_proba[0][1] * 100

    return jsonify({'attrition_probability': attrition_probability})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard-view')
def dashboard_view():
    return render_template('dashboard_view.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
