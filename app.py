# app.py
from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import requests
import io
import os
from datetime import datetime

app = Flask(__name__)
OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ImageUpscaler:
    @staticmethod
    def download_image(url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        image_data = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    @staticmethod
    def upscale_image(image, scale_factor):
        return cv2.resize(image, None, 
                         fx=scale_factor, 
                         fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        image_url = request.form['image_url']
        scale_factor = float(request.form['scale_factor'])
        
        upscaler = ImageUpscaler()
        image = upscaler.download_image(image_url)
        upscaled = upscaler.upscale_image(image, scale_factor)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_DIR}/upscaled_{timestamp}.jpg"
        cv2.imwrite(output_path, upscaled)
        
        return jsonify({
            'success': True,
            'image_path': output_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
    