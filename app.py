# app.py
from main import get_api_url
from flask import Flask, render_template, request, redirect, url_for
import requests
import os
import cv2
import numpy as np

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder untuk upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Tidak ada file yang diunggah", 400
    file = request.files['file']
    if file.filename == '':
        return "Tidak ada file yang dipilih", 400
    if file:
        # Baca file dari request tanpa menyimpan ke disk
        file_stream = file.stream.read()  # Membaca file ke dalam stream
        file_array = np.frombuffer(file_stream, np.uint8)  # Membuat array numpy
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)  # Decode ke format gambar

        # Panggil fungsi get_api_url dengan gambar langsung
        api_url = get_api_url(image)  # Modifikasi get_api_url agar menerima gambar langsung
        # Arahkan pengguna ke halaman /index dengan api_url sebagai query string
        return redirect(url_for('use_api_url', api_url=api_url))
    
@app.route('/index')
def use_api_url():
    api_url = request.args.get('api_url', '')

    if not api_url:
        return "URL API tidak ditemukan.", 400
    try:
        # Fetch data dari API
        response = requests.get(api_url)
        response.raise_for_status()  # Pastikan tidak ada error dalam request
        api_data = response.json()  # Jika respon berbentuk JSON
        
        # Ambil data spesifik dari respon API
        title = api_data.get("title", "Title tidak tersedia")
        hadeeth = api_data.get("hadeeth", "Title tidak tersedia")
        attribution = api_data.get("attribution", "Title tidak tersedia")
        grade = api_data.get("grade", "Title tidak tersedia")
        explanation = api_data.get("explanation", "Title tidak tersedia")
        hadeeth_ar = api_data.get("hadeeth_ar", "Title tidak tersedia")
        # description = api_data.get("description", "Deskripsi tidak tersedia")

    except requests.exceptions.RequestException as e:
        title = "Error"
        hadeeth = "Error"
        attribution = "Error"
        grade = "Error"
        explanation = "Error"
        hadeeth_ar = "Error"
    except Exception as e:
        title = "Error"
        hadeeth = "Error"
        attribution = "Error"
        grade = "Error"
        explanation = "Error"
        hadeeth_ar = "Error"
    # Render ke template
    return render_template('index.html', 
                           title=title,
                           hadeeth=hadeeth,
                           hadeeth_ar=hadeeth_ar,
                           attribution=attribution,
                           grade=grade,
                           explanation=explanation )

    # print(f"API URL yang didapatkan: {api_url}")
    # Gunakan api_url sesuai kebutuhan, misalnya untuk request
    # response = requests.get(api_url)
    # print(response.text)

if __name__ == "__main__":
    app.run(debug=True)
