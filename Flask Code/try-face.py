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
    test_data_dir = get_test_data_path()

    # Get the full path
    smile_img = os.path.join(test_data_dir, "smileS.jpg")
    disgusted_img = os.path.join(test_data_dir, "disgusedS.jpg")
    surprised_img = os.path.join(test_data_dir, "surprisedS.jpg")

    smile_face_prediction = detector.detect_image(smile_img)
    disgusted_face_prediction = detector.detect_image(disgusted_img)
    surprised_face_prediction = detector.detect_image(surprised_img)

    # Extract relevant AUs from the predictions
    smile_AU01 = smile_face_prediction.aus['AU01'][0]
    smile_AU06 = smile_face_prediction.aus['AU06'][0]
    smile_AU12 = smile_face_prediction.aus['AU12'][0]
    disgusted_AU04 = disgusted_face_prediction.aus['AU04'][0]
    disgusted_AU07 = disgusted_face_prediction.aus['AU07'][0]
    disgusted_AU09 = disgusted_face_prediction.aus['AU09'][0]
    surprised_AU01 = surprised_face_prediction.aus['AU01'][0]
    surprised_AU02 = surprised_face_prediction.aus['AU02'][0]
    surprised_AU04 = surprised_face_prediction.aus['AU04'][0]

    # Create the AUResult array
    AUResult = [smile_AU01, smile_AU06, smile_AU12, disgusted_AU04, disgusted_AU07, disgusted_AU09, surprised_AU01, surprised_AU02, surprised_AU04]

    # Print the results
    print("AUResult: ", AUResult)

    result = loaded_model.predict(np.array(AUResult).reshape(1,-1))
    
    # Return the results as JSON
    return jsonify({"message": result})

# if __name__ == '__main__':
#     app.run(debug=True, port=8000, host='0.0.0.0')