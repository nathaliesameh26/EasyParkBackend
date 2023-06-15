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
model_path = 'parkinson_disease_detection1.h5'
loaded_model = tf.keras.models.load_model(model_path)
@app.route('/upload', methods=['POST'])
def upload():
    

    if(request.method=="POST"):
        imagefile= request.files['image']
        filename=werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/"+filename)
        input_image = Image.open("./uploadedimages/"+filename).convert("RGB")
#         train_data_generator = ImageDataGenerator(rotation_range=360, 
#                                     width_shift_range=0.0, 
#                                     height_shift_range=0.0, 
# #                                     brightness_range=[0.5, 1.5],
#                                     horizontal_flip=True, 
#                                     vertical_flip=True)
#         x = list(x_train)
#         y = list(y_train)

        
#         x_aug_train = []
#         y_aug_train = []
    
#         for (i, v) in enumerate(y):
#             x_img = x[i]
#             x_img = np.array(x_img)
#             x_img = np.expand_dims(x_img, axis=0)
#             aug_iter = train_data_generator.flow(x_img, batch_size=1, shuffle=True)
#             for j in range(70):
#                 aug_image = next(aug_iter)[0].astype('uint8')
#                 x_aug_train.append(aug_image)
#                 y_aug_train.append(v)
#         print(len(x_aug_train))
#         print(len(y_aug_train))
#         x_train = x + x_aug_train
#         y_train = y + y_aug_train
#         print(len(x_train))
#         print(len(y_train))

#         test_data_generator = ImageDataGenerator(rotation_range=360, 
#                                     width_shift_range=0.0, 
#                                     height_shift_range=0.0, 
#                                     brightness_range=[0.5, 1.5],
#                                     horizontal_flip=True, 
#                                     vertical_flip=True)
        
#         x = list(x_test)
#         y = list(y_test)
        
#         x_aug_test = []
#         y_aug_test = []

#         for (i, v) in enumerate(y):
#             x_img = x[i]
#             x_img = np.array(x_img)
#             x_img = np.expand_dims(x_img, axis=0)
#             aug_iter = test_data_generator.flow(x_img, batch_size=1, shuffle=True)
#             for j in range(20):
#                 aug_image = next(aug_iter)[0].astype('uint8')
#                 x_aug_test.append(aug_image)
#                 y_aug_test.append(v)
#         print(len(x_aug_test))
#         print(len(y_aug_test))     

#         x_test = x + x_aug_test
#         y_test = y + y_aug_test
#         print(len(x_test))
#         print(len(y_test))   
        
        # for i in range(len(x_train)):
        #     img = x_train[i]
        #     img = cv2.resize(img, (128, 128))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     x_train[i] = img

        # for i in range(len(x_test)):
        #     img = x_test[i]
        #     img = cv2.resize(img, (128, 128))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     x_test[i] = img  

        # x_train = np.array(x_train)
        # x_test = np.array(x_test)  

        # x_train = x_train/255.0
        # x_test = x_test/255.0    

        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(y_train)
        # print(y_train.shape)

        # label_encoder = LabelEncoder()
        # y_test = label_encoder.fit_transform(y_test)
        # print(y_test.shape)

        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

        # x_train = np.expand_dims(x_train, axis=-1)
        # x_test = np.expand_dims(x_test, axis=-1)

        
        # print(x_train.shape)
        # print(y_train.shape)
        # print(x_test.shape)
        # print(y_test.shape)

    
        labels = ['Healthy', 'Parkinson']
        # image_healthy = plt.imread('uploadedimages\image_picker105999189885744963.png')
        # image_parkinson = plt.imread('uploadedimages\image_picker105999189885744963.png')
        image_healthy = cv2.imread('./uploadedimages/image_picker70939151257986256.png')
        image_parkinson = cv2.imread('./uploadedimages/image_picker70939151257986256.png')
        
        image_healthy = cv2.resize(image_healthy, (128, 128))
        image_healthy = cv2.cvtColor(image_healthy, cv2.COLOR_BGR2GRAY)
        image_healthy = np.array(image_healthy)
        image_healthy = np.expand_dims(image_healthy, axis=0)
        image_healthy = np.expand_dims(image_healthy, axis=-1)
        
        image_parkinson = cv2.resize(image_parkinson, (128, 128))
        image_parkinson = cv2.cvtColor(image_parkinson, cv2.COLOR_BGR2GRAY)
        image_parkinson = np.array(image_parkinson)
        image_parkinson = np.expand_dims(image_parkinson, axis=0)
        image_parkinson = np.expand_dims(image_parkinson, axis=-1)  

        ypred_healthy = loaded_model.predict(np.array(image_healthy).tolist()).tolist()
        ypred_parkinson = loaded_model.predict(np.array(image_parkinson).tolist()).tolist()
        ypred_healthy=labels[np.argmax(ypred_healthy[0], axis=0)]


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

# if __name__=="__main__":
#     app.run(debug=True,port=8000, host='0.0.0.0')