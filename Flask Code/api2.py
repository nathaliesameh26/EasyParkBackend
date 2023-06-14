
from flask import Flask,request,jsonify
import werkzeug
from PIL import Image
import h5py    
import numpy as np
import cv2 
from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.regularizers import l2
import tensorflow as tf
import os
import pandas as pd
# from feat import Detector
# from feat.utils.io import get_test_data_path
# from feat.plotting import imshow



# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# model = h5py.File('parkinson_disease_detectionWave.h5','r')
# ls = list(model.keys())
app = Flask(__name__)
model_path = 'cnn_wave_model.h5'
loaded_model = tf.keras.models.load_model(model_path)
@app.route('/upload', methods=['POST'])
def upload():
    if(request.method=="POST"):
        imagefile= request.files['image']
        filename=werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/"+filename)
        input_image = Image.open("./uploadedimages/"+filename).convert("RGB")

    
        labels = ['Healthy', 'Parkinson']
        # image_healthy = plt.imread('uploadedimages\image_picker105999189885744963.png')
        # image_parkinson = plt.imread('uploadedimages\image_picker105999189885744963.png')
        # image_healthy = cv2.imread('./uploadedimages/WaveP.png')
        # image_healthy = cv2.imread('./uploadedimages/wave.jpg')[..., ::-1]
        image_healthy = np.array(input_image)
        # image_parkinson = cv2.imread('./uploadedimages/WaveP.png')
        
        # image_healthy = cv2.resize(image_healthy, (256, 512))
        image_healthy = cv2.resize(image_healthy, (512, 256))
        image_healthy = np.array(image_healthy) / 255
        image_healthy = np.transpose(image_healthy, (1, 0, 2))
        image_healthy = np.expand_dims(image_healthy, axis=0)

        ypred_healthy = loaded_model.predict(image_healthy)
        ypred_healthy = labels[np.argmax(ypred_healthy)]
        # image_healthy = cv2.cvtColor(image_healthy, cv2.COLOR_BGR2GRAY)
        # image_healthy = np.array(image_healthy)
        # image_healthy = np.expand_dims(image_healthy, axis=0)
        # image_healthy = np.expand_dims(image_healthy, axis=-1)
        
        # image_parkinson = cv2.resize(image_parkinson, (256, 512))
        # # image_parkinson = cv2.cvtColor(image_parkinson, cv2.COLOR_BGR2GRAY)
        # image_parkinson = np.array(image_parkinson)
        # image_parkinson = np.expand_dims(image_parkinson, axis=0)
        # image_parkinson = np.expand_dims(image_parkinson, axis=-1)  

        # ypred_healthy = loaded_model.predict(np.array(image_healthy).tolist()).tolist()
        # ypred_parkinson = loaded_model.predict(np.array(image_parkinson).tolist()).tolist()
        # ypred_healthy=labels[np.argmax(ypred_healthy[0], axis=0)]


        # figure = plt.figure(figsize=(2, 2))
        # img_healthy = np.squeeze(image_healthy, axis=0)
        # plt.imshow(img_healthy)
        # plt.axis('off')
        # plt.title(f'Prediction by the model: {labels[np.argmax(ypred_healthy[0], axis=0)]}')
        # plt.show()

        # figure = plt.figure(figsize=(2, 2))
        # image_parkinson = np.squeeze(image_parkinson, axis=0)
        # plt.imshow(image_parkinson)
        # plt.axis('off')
        # plt.title(f'Prediction by the model: {labels[np.argmax(ypred_parkinson[0], axis=0)]}')
        # plt.show()

        # !deepCC parkinson_disease_detection.h5
        # loaded_model.close()

        
        return jsonify({
        "message": ypred_healthy
        })

if __name__ =="__main__":
    app.run(debug=True,port=8000,host='0.0.0.0')