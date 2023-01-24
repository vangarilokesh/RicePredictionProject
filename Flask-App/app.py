from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from keras.models import model_from_json
from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# im=im.resize((300,300))
# im=np.expand_dims(im,axis=0)
# pred=loaded_model.model.predict(im)
# print(pred)   # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
# model.save('')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='sparse_categorical_crossentropy',
                     optimizer="adam",
                     metrics=['accuracy'])
print("Loaded model from disk")
print('Model loaded. Check http://127.0.0.1:5000/')

'''from matplotlib.pyplot import imshow
def Prediction(img):
    x = np.array(img.resize((32,32)))
    x = x.reshape(1,32,32,1)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)
    print(str(res[0][classification]*100) + '% Confidence ' + dicti[classification])'''


def model_predict(img_path, model):
    # img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    # x = image.im_to_array(img)
    im = Image.open(img_path)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    x = np.array(im.resize((32, 32)).convert('L'))
    x = x.reshape(1, 32, 32, 1)
    #im = np.expand_dims(im, axis=0)
    pred = model.predict(x)
    print(pred)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    # preds = model.predict(im)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        #print("Before predict call")
        preds = model_predict(file_path, loaded_model)
        #print("After predict call")
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        index = np.where(preds == np.amax(preds))[1][0]
        arr = {0: 'Arborio', 1: 'Basmati',
               2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}
        print(preds[0][index])
        result = arr[index]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
