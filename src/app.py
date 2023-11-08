import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask("Neuralnsight",static_folder="static",template_folder="templates")
model =load_model('Model/BrainTumorModel_10epochs_Categorical.h5')


print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(predictions):
    if predictions > 0.5:
        return "Positive Brain Tumor Detected"
    else:
        return "No Brain Tumor Detected"



def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class



@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')


@app.route('/model', methods=['GET'])
def model_page():
    return render_template('model_playground.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('home.html#contact')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)