# app.py
from main import get_api_url
from flask import Flask, render_template, request, redirect, url_for
import requests
import os
import cv2
import numpy as np

app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

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
        file_stream = file.stream.read()  
        file_array = np.frombuffer(file_stream, np.uint8)  
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR) 
        api_url = get_api_url(image)  
        return redirect(url_for('use_api_url', api_url=api_url))
    
@app.route('/index')
def use_api_url():
    api_url = request.args.get('api_url', '')

    if not api_url:
        return "URL API tidak ditemukan.", 400
    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        api_data = response.json()  

        title = api_data.get("title", "Title tidak tersedia")
        hadeeth = api_data.get("hadeeth", "Title tidak tersedia")
        attribution = api_data.get("attribution", "Title tidak tersedia")
        grade = api_data.get("grade", "Title tidak tersedia")
        explanation = api_data.get("explanation", "Title tidak tersedia")
        hadeeth_ar = api_data.get("hadeeth_ar", "Title tidak tersedia")
      
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
    return render_template('index.html', 
                           title=title,
                           hadeeth=hadeeth,
                           hadeeth_ar=hadeeth_ar,
                           attribution=attribution,
                           grade=grade,
                           explanation=explanation )
if __name__ == "__main__":
    app.run(debug=True)
