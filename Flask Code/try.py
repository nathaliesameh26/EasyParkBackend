import pickle
from flask import Flask, request, jsonify
import werkzeug
from PIL import Image
import numpy as np
import tensorflow as tf
from feat import Detector
import os

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="xgb",
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

app = Flask(__name__)
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == "POST":
        smile_image = request.files['smile_image']
        disgusted_image = request.files['disgusted_image']
        surprised_image = request.files['surprised_image']
        
        smile_filename = werkzeug.utils.secure_filename(smile_image.filename)
        disgusted_filename = werkzeug.utils.secure_filename(disgusted_image.filename)
        surprised_filename = werkzeug.utils.secure_filename(surprised_image.filename)
        
        smile_image.save("uploadedimages/" + smile_filename)
        disgusted_image.save("uploadedimages/" + disgusted_filename)
        surprised_image.save("uploadedimages/" + surprised_filename)
        
    try:
        # Save the images
            smile_image.save("uploadedimages/" + smile_filename)
            disgusted_image.save("uploadedimages/" + disgusted_filename)
            surprised_image.save("uploadedimages/" + surprised_filename)

        # Open the images using file paths
            smile_input_image = Image.open("uploadedimages/" + smile_filename)
            disgusted_input_image = Image.open("uploadedimages/" + disgusted_filename)
            surprised_input_image = Image.open("uploadedimages/" + surprised_filename)

            # Initialize the results list
            results = []

            # Detect the facial features and extract the relevant AUs for each image
            smile_face_prediction = detector.detect_image(smile_input_image).get_subface('Smile')
            disgusted_face_prediction = detector.detect_image(disgusted_input_image).get_subface('Disgusted')
            surprised_face_prediction = detector.detect_image(surprised_input_image).get_subface('Surprised')

            smile_AU01 = smile_face_prediction.aus['AU01'][0]
            smile_AU06 = smile_face_prediction.aus['AU06'][0]
            smile_AU12 = smile_face_prediction.aus['AU12'][0]
            disgusted_AU04 = disgusted_face_prediction.aus['AU04'][0]
            disgusted_AU07 = disgusted_face_prediction.aus['AU07'][0]
            disgusted_AU09 = disgusted_face_prediction.aus['AU09'][0]
            surprised_AU01 = surprised_face_prediction.aus['AU01'][0]
            surprised_AU02 = surprised_face_prediction.aus['AU02'][0]
            surprised_AU04 = surprised_face_prediction.aus['AU04'][0]

            # Create the AUs list
            AUs = [smile_AU01, smile_AU06, smile_AU12, disgusted_AU04, disgusted_AU07, disgusted_AU09,
                   surprised_AU01, surprised_AU02, surprised_AU04]

            # Reshape the AUs and predict the label
            AUs_array = np.array(AUs).reshape(1, -1)
            pred_label = loaded_model.predict(AUs_array)[0]

            # Add the result to the list
            result = {
                'smile_image_path': smile_filename,
                'disgusted_image_path': disgusted_filename,
                'surprised_image_path': surprised_filename,
                'action_units': AUs,
                'label': 'Parkinson' if pred_label == 1 else 'Not Parkinson'
            }
            results.append(result)
        
    except Exception as e:
            result = {'error': str(e)}
            results.append(result)
   
    # Return the results as JSON
    return jsonify({
        "message": results
        # "message2": ypred_parkinson
        })

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')