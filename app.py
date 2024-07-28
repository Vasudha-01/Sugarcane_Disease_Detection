from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.color import rgb2yuv, yuv2rgb
from scipy.signal import convolve2d

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Save uploads to static/uploads
app.config['SECRET_KEY'] = 'supersecretkey'

# Load the model
model = load_model('models/corrected_model.h5')

# Define a sharpen filter
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary='fill', fillvalue=0)
    return image

def convolver_rgb(image, kernel, iterations=1):
    img_yuv = rgb2yuv(image)
    img_yuv[:, :, 0] = multi_convolver(img_yuv[:, :, 0], kernel, iterations)
    final_image = yuv2rgb(img_yuv)
    return final_image

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    img_array = convolver_rgb(img_array[0], sharpen, iterations=1)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    classes = ['Healthy', 'Red Rot', 'Red Rust']
    return classes[np.argmax(prediction)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file to the static/uploads directory
        file.save(file_path)
        
        result = predict_disease(file_path)
        
        return render_template('result.html', result=result, filename=filename)

if __name__ == '__main__':
    # Create the uploads directory inside static if it doesn't exist
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
