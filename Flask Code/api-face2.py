import os
import pickle
from flask import Flask, jsonify, request
import cv2
from feat import Detector
import numpy as np
from werkzeug.utils import secure_filename
from feat.utils.io import get_test_data_path

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

app = Flask(__name__)

# Load the trained model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'image_paths' not in request.files:
        return jsonify({'error': 'Invalid request. Missing image_paths parameter.'}), 400

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('image_paths')

    # Initialize the results list
    results = []

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Save the uploaded file
            filename = secure_filename(uploaded_file.filename)
            uploaded_image_path = os.path.join("uploadedimages", filename)
            uploaded_file.save(uploaded_image_path)

            # Load the smile image using OpenCV
            smile_img_path = os.path.join("uploadedimages", "smileS.jpg")
            uploaded_file.save(smile_img_path)
            imgSmile = cv2.imread(smile_img_path)

            # Detect the facial features and extract the relevant AUs for smile
            face_prediction_smile = detector.detect_image(imgSmile)
            smile_face_prediction = face_prediction_smile.get_subface('Smile')
            smile_AU01 = smile_face_prediction.aus['AU01'][0]
            smile_AU06 = smile_face_prediction.aus['AU06'][0]
            smile_AU12 = smile_face_prediction.aus['AU12'][0]

            # Load the disgusted image using OpenCV
            disgusted_img_path = os.path.join("uploadedimages", "disgustedS.jpg")
            uploaded_file.save(disgusted_img_path)
            imgDisgusted = cv2.imread(disgusted_img_path)

            # Detect the facial features and extract the relevant AUs for disgusted
            face_prediction_disgusted = detector.detect_image(imgDisgusted)
            disgusted_face_prediction = face_prediction_disgusted.get_subface('Disgusted')
            disgusted_AU04 = disgusted_face_prediction.aus['AU04'][0]
            disgusted_AU07 = disgusted_face_prediction.aus['AU07'][0]
            disgusted_AU09 = disgusted_face_prediction.aus['AU09'][0]

            # Load the surprised image using OpenCV
            surprised_img_path = os.path.join("uploadedimages", "surprisedS.jpg")
            uploaded_file.save(surprised_img_path)
            imgSurprised = cv2.imread(surprised_img_path)

            # Detect the facial features and extract the relevant AUs for surprised
            face_prediction_surprised = detector.detect_image(imgSurprised)
            surprised_face_prediction = face_prediction_surprised.get_subface('Surprised')
            surprised_AU01 = surprised_face_prediction.aus['AU01'][0]
            surprised_AU02 = surprised_face_prediction.aus['AU02'][0]
            surprised_AU04 = surprised_face_prediction.aus['AU04'][0]

            # Create the AUs list
            AUs = [smile_AU01, smile_AU06, smile_AU12, disgusted_AU04, disgusted_AU07, disgusted_AU09, surprised_AU01, surprised_AU02, surprised_AU04]

            # Reshape the AUs and predict the label
            AUs_array = np.array(AUs).reshape(1, -1)
            pred_label = loaded_model.predict(AUs_array)[0]

            # Add the result to the list
            result = {'image_path': filename, 'action_units': AUs, 'label': 'Parkinson' if pred_label == 1 else 'Not Parkinson'}
            results.append(result)
        except Exception as e:
            result = {'image_path': filename, 'error': str(e)}
            results.append(result)

    # Return the results as JSON
    return jsonify({"message": results})

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
























# #THIS IS THE WORKINGG ONEEEEEEE

# # import json
# # import os
# # import pickle
# # from flask import Flask, jsonify, request
# # import cv2
# # from feat import Detector
# # import numpy as np
# # import werkzeug
# # from werkzeug.utils import secure_filename

# # detector = Detector(
# #     face_model="retinaface",
# #     landmark_model="mobilefacenet",
# #     au_model='xgb',
# #     emotion_model="resmasknet",
# #     facepose_model="img2pose",
# # )

# # app = Flask(__name__)

# # # Load the trained model
# # filename = 'finalized_model.sav'
# # loaded_model = pickle.load(open(filename, 'rb'))

# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     if 'image_paths' not in request.files:
# #         return jsonify({'error': 'Invalid request. Missing image_paths parameter.'}), 400

# #     # Get the list of uploaded files
# #     uploaded_files = request.files.getlist('image_paths')

# #     # Initialize the results list
# #     results = []

# #     # Process each uploaded file
# #     for uploaded_file in uploaded_files:
# #         try:
# #             # Save the uploaded file
# #             filename = secure_filename(uploaded_file.filename)
# #             uploaded_image_path = os.path.join('uploadedimages', filename)
# #             uploaded_file.save(uploaded_image_path)

# #             # Load the image
# #             img = cv2.imread(uploaded_image_path)

# #             # Detect the facial features and extract the relevant AUs
# #             face_prediction = detector.detect_image(img)
# #             smile_face_prediction = face_prediction.get_subface('Smile')
# #             disgusted_face_prediction = face_prediction.get_subface('Disgusted')
# #             surprised_face_prediction = face_prediction.get_subface('Surprised')
# #             smile_AU01 = smile_face_prediction.aus['AU01'][0]
# #             smile_AU06 = smile_face_prediction.aus['AU06'][0]
# #             smile_AU12 = smile_face_prediction.aus['AU12'][0]
# #             disgusted_AU04 = disgusted_face_prediction.aus['AU04'][0]
# #             disgusted_AU07 = disgusted_face_prediction.aus['AU07'][0]
# #             disgusted_AU09 = disgusted_face_prediction.aus['AU09'][0]
# #             surprised_AU01 = surprised_face_prediction.aus['AU01'][0]
# #             surprised_AU02 = surprised_face_prediction.aus['AU02'][0]
# #             surprised_AU04 = surprised_face_prediction.aus['AU04'][0]

# #             # Create the AUs list
# #             AUs = [smile_AU01, smile_AU06, smile_AU12, disgusted_AU04, disgusted_AU07, disgusted_AU09, surprised_AU01, surprised_AU02, surprised_AU04]

# #             # Reshape the AUs and predict the label
# #             AUs_array = np.array(AUs).reshape(1, -1)
# #             pred_label = loaded_model.predict(AUs_array)[0]

# #             # Add the result to the list
# #             result = {'image_path': filename, 'action_units': AUs, 'label': 'Parkinson' if pred_label == 1 else 'Not Parkinson'}
# #             results.append(result)
# #         except Exception as e:
# #             # Handle any errors that occur during image processing
# #             result = {'image_path': filename, 'error': str(e)}
# #             results.append(result)

# #     # Return the results as JSON
# #     return jsonify(results)

# # if __name__ == '__main__':
# #     app.run(debug=True, port=8000, host='0.0.0.0')



















# #---------------------------------------------------------------------------

# # import json
# # import pickle
# # from flask import Flask, jsonify, request
# # import pandas as pd
# # import numpy as np
# # from imblearn.over_sampling import SMOTE
# # from sklearn.preprocessing import scale
# # from sklearn import svm
# # from sklearn.model_selection import cross_val_predict
# # # from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
# # import cv2
# # from feat import Detector
# # # import feat
# # # from feat.detector import Detector
# # # from feat.utils import openface_2d_landmark_columns

# # detector = Detector(
# #     face_model="retinaface",
# #     landmark_model="mobilefacenet",
# #     au_model='xgb',
# #     emotion_model="resmasknet",
# #     facepose_model="img2pose",
# # )
# # detector

# # app = Flask(__name__)

# # # Load the trained model
# # filename = 'finalized_model.sav'
# # loaded_model = pickle.load(open(filename, 'rb'))

# # # Define the feature names
# # feats = ['AU_01_t12', 'AU_06_t12', 'AU_12_t12', 'AU_04_t13', 'AU_07_t13', 'AU_09_t13', 'AU_01_t14', 'AU_02_t14', 'AU_04_t14']

# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     # Get the request data
# #     # data = request.json
# #     img_paths = []
# #     img_paths = json.loads(request.form.get('image_paths'))
# #     # Get the file paths
# #     #img_paths = data['/uploadedimages/smileS.jpg','/uploadedimages/disgustedS.jpg''/uploadedimages/surprisedS.jpg']
# #     # Initialize the results list
# #     results = []
    
# #     # Loop through the images and predict the label for each
# #     AUs_list = []
# #     for img_path in img_paths:
# #         # Load the image
# #         img = cv2.imread(img_path)
        
# #         # Detect the facial features and extract the relevant AUs
# #         face_prediction = detector.detect_image(img)
# #         smile_face_prediction = face_prediction.get_subface('Smile')
# #         disgusted_face_prediction = face_prediction.get_subface('Disgusted')
# #         surprised_face_prediction = face_prediction.get_subface('Surprised')
# #         smile_AU01 = smile_face_prediction.aus['AU01'][0]
# #         smile_AU06 = smile_face_prediction.aus['AU06'][0]
# #         smile_AU12 = smile_face_prediction.aus['AU12'][0]
# #         disgusted_AU04 = disgusted_face_prediction.aus['AU04'][0]
# #         disgusted_AU07 = disgusted_face_prediction.aus['AU07'][0]
# #         disgusted_AU09 = disgusted_face_prediction.aus['AU09'][0]
# #         surprised_AU01 = surprised_face_prediction.aus['AU01'][0]
# #         surprised_AU02 = surprised_face_prediction.aus['AU02'][0]
# #         surprised_AU04 = surprised_face_prediction.aus['AU04'][0]
# #         AUs = [smile_AU01, smile_AU06, smile_AU12, disgusted_AU04, disgusted_AU07, disgusted_AU09, surprised_AU01, surprised_AU02, surprised_AU04]
# #         AUs_list.extend(AUs)
    
# #     # Reshape the AUs and predict the label
# #     AUs_array = np.array(AUs_list).reshape(1, -1)
# #     pred_label = loaded_model.predict(AUs_array)[0]
        
# #     # Add the result to the list
# #     results.append({'action_units': AUs_list, 'label': 'Parkinson' if pred_label == 1 else 'Not Parkinson'})
    
# #     # Print the AUs
# #     print("AUs: ", AUs_list)
# #     print("Predicted label: ", pred_label)
    
# #     # Return the results as JSON
# #     return jsonify(results)

# # if __name__ == '__main__':
# #     app.run(debug=True,port=8000,host='0.0.0.0')



# #---------------------------------------------------------------------------------


# # # from flask import Flask,request,jsonify
# # # import werkzeug
# # # from PIL import Image
# # # import h5py    
# # # import numpy as np
# # # import cv2 
# # # from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D
# # # from keras.models import Model, Sequential
# # # from keras.initializers import glorot_uniform
# # # from keras.optimizers import Adam, SGD
# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import classification_report
# # # from keras.preprocessing.image import ImageDataGenerator
# # # from sklearn.model_selection import train_test_split
# # # from keras.utils import to_categorical
# # # from sklearn.preprocessing import LabelEncoder
# # # from sklearn.metrics import classification_report, confusion_matrix
# # # import seaborn as sns
# # # from keras.regularizers import l2
# # # import tensorflow as tf
# # # import os
# # # import pandas as pd

# # # app = Flask(__name__)
# # # model_path = 'finalized_model.sav'
# # # loaded_model = tf.keras.models.load_model(model_path)
# # # @app.route('/upload', methods=['POST'])
# # # def upload():
# # #     if(request.method=="POST"):
# # #         imagefile= request.files['image']
# # #         filename=werkzeug.utils.secure_filename(imagefile.filename)
# # #         imagefile.save("uploadedimages/"+filename)
# # #         input_image = Image.open("uploadedimages/"+filename)
# # #         input_image = Image.open("uploadedimages/"+filename)
# # #         input_image = Image.open("uploadedimages/"+filename)

# # #         labels = ['Healthy', 'Parkinson']
# # #         # image_healthy = plt.imread('uploadedimages/SpiralP1.png')
# # #         # image_parkinson = plt.imread('uploadedimages/SpiralP1.png')
# # #         image_healthy = np.array(input_image)
# # #         image_parkinson = np.array(input_image)

# # #         image_healthy = cv2.resize(image_healthy, (128, 128))
# # #       # Extract relevant AUs from the predictions
# # #         smile_AU01 = smile_face_prediction.aus['AU01'][0]
# # #         smile_AU06 = smile_face_prediction.aus['AU06'][0]
# # #         smile_AU12 = smile_face_prediction.aus['AU12'][0]
# # #         disgusted_AU04 = disgusted_face_prediction.aus['AU04'][0]
# # #         disgusted_AU07 = disgusted_face_prediction.aus['AU07'][0]
# # #         disgusted_AU09 = disgusted_face_prediction.aus['AU09'][0]
# # #         surprised_AU01 = surprised_face_prediction.aus['AU01'][0]
# # #         surprised_AU02 = surprised_face_prediction.aus['AU02'][0]
# # #         surprised_AU04 = surprised_face_prediction.aus['AU04'][0]

# # #         # Create the AUResult array
# # #         AUResult = [smile_AU01, smile_AU06, smile_AU12, disgusted_AU04, disgusted_AU07, disgusted_AU09, surprised_AU01, surprised_AU02, surprised_AU04]

# # #         # load the model from disk
# # #         loaded_model = pickle.load(open(filename, 'rb'))
# # #         result = loaded_model.predict(np.array(AUResult).reshape(1,-1))
# # #         print(result)


# # #         return jsonify({
# # #         "message": ypred_healthy
        
# # #         })

# # # if __name__ =="__main__":
# # #     app.run(debug=True,port=8000,host='0.0.0.0')