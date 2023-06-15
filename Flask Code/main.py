
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
import pickle
from flask import Flask, jsonify, request
from feat import Detector
from werkzeug.utils import secure_filename
from feat.utils.io import get_test_data_path
import pickle
from flask import Flask, jsonify, request
import parselmouth
from parselmouth.praat import call
import csv
from keras.models import load_model
import tensorflow as tf
from pydub import AudioSegment
import nolds
import base64
# from dotenv import load_dotenv
# load_dotenv()



################################     SPIRAL     ######################################### 
app = Flask(__name__)
@app.route('/uploadSpiral', methods=['POST'])
def uploadSpiral():
    model_path = 'parkinson_disease_detection1.h5'
    loaded_model = tf.keras.models.load_model(model_path)
    if(request.method=="POST"):
        imagefile= request.files['image']
        filename=werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("uploadedimages/"+filename)
        input_image = Image.open("uploadedimages/"+filename)
        labels = ['Healthy', 'Parkinson']
        image_healthy = np.array(input_image)
        image_parkinson = np.array(input_image)
        
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
        ypred_parkinson=labels[np.argmax(ypred_parkinson[0], axis=0)]
        
        return jsonify({
        "message": ypred_healthy
        })


@app.route('/uploadWave', methods=['POST'])
def uploadWave():
    model_path = 'cnn_wave_model.h5'
    loaded_model = tf.keras.models.load_model(model_path)
    if(request.method=="POST"):
        imagefile= request.files['image']
        filename=werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/"+filename)
        input_image = Image.open("./uploadedimages/"+filename).convert("RGB")

        labels = ['Healthy', 'Parkinson']
        image_healthy = np.array(input_image)
        image_healthy = cv2.resize(image_healthy, (512, 256))
        image_healthy = np.array(image_healthy) / 255
        image_healthy = np.transpose(image_healthy, (1, 0, 2))
        image_healthy = np.expand_dims(image_healthy, axis=0)

        ypred_healthy = loaded_model.predict(image_healthy)
        ypred_healthy = labels[np.argmax(ypred_healthy)]
       

        
        return jsonify({
        "message": ypred_healthy
        })  


@app.route('/uploadFace', methods=['POST'])
def uploadFace():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)
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
 

@app.route('/process_audio', methods=['POST'])
def process_audio():
    with open('content_model6xg.pkl', 'rb') as file:
         model = pickle.load(file)
    # Load audio file from request
    audio_file = request.form['audio']

    # # Save the audio file temporarily
    # audio_path = '/testtt.wav'
    # audio_file.save(audio_path)

    # audio_data = request.form['testtt.wav']
    decoded_audio = base64.b64decode(audio_file)

    # Save the audio file
    audio_filename = 'audio.wav'  # Specify the desired filename and extension
    audio_filepath = os.path.join('C:/Users/amira/Desktop/Graduation Project/voice/',audio_filename)
    
    with open(audio_filepath, 'wb') as audio_file:
        audio_file.write(decoded_audio)

    # Extract raw audio data and sample rate
    audio_array = np.array(AudioSegment.from_wav(audio_filepath).get_array_of_samples())
    sample_rate = AudioSegment.from_wav(audio_filepath).frame_rate


    # Process the audio data
    # (Place the existing code here)

    # Create a list of headers for the CSV file
    headers = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
               'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
               'MDVP:APQ', 'Shimmer:DDA', 'HNR', 'DFA', 'spread1', 'spread2', 'PPE']
    # Calculate the FFT of the audio signal
    fft = np.fft.fft(audio_array)

    # Calculate the frequency values corresponding to each point in the FFT
    freqs = np.fft.fftfreq(len(audio_array), 1.0 / sample_rate)

    # Find the indices of the FFT array corresponding to the fundamental frequency range of interest
    fundamental_range = (10, 500)
    fundamental_freqs = np.where((freqs >= fundamental_range[0]) & (freqs <= fundamental_range[1]))

    # Take the absolute value of the FFT array and extract the magnitudes corresponding to the fundamental frequency range
    fft_mags = np.abs(fft[fundamental_freqs])

    # Find the index of the maximum magnitude in the FFT array
    max_index = np.argmax(fft_mags)
    # Calculate the frequency values corresponding to the maximum and minimum indices
    max_freq = freqs[fundamental_freqs][max_index]
    min_freq = freqs[fundamental_freqs][0]

    # Calculate the average frequency value by taking the mean of the frequency values in the fundamental range
    avg_freq = np.mean(freqs[fundamental_freqs])

    # Calculate jitter, shimmer, hnr, RAP, PPQ, DDP, and MDVP:APQ
    sound = parselmouth.Sound(audio_filepath)


    pitch = sound.to_pitch()
    f0_values = pitch.selected_array['frequency']

    # Calculate the DFA of the audio signal
    dfa = nolds.dfa(f0_values)

    # Calculate the spread1 and spread2 measures of fundamental frequency variation
    spread1 = np.std(f0_values)
    spread2 = np.ptp(f0_values)

# Calculate the PPE measure of fundamental frequency variation
    ppe = nolds.sampen(f0_values)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 300)
    jitter_percent = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    pitch = sound.to_pitch()
    diff = np.abs(pitch.selected_array['frequency'][1:] - pitch.selected_array['frequency'][:-1])
    mean_abs_diff = np.mean(diff)
    std_dev_diff = np.std(diff)
    mdvp_apq = (std_dev_diff / mean_abs_diff) / 100
    # Create a list of the values to be printed
    values = [avg_freq, max_freq, min_freq, jitter_percent, jitter_absolute, rap, ppq, ddp, shimmer, shimmer_db,
              shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, hnr, dfa, spread1, spread2, ppe]

    # Create a new CSV file and write the headers
    csv_path = 'output.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # Write the values to the file
        writer.writerow(values)

    # Load the CSV file and prepare the data for prediction
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

    # Perform any necessary preprocessing on the data
    preprocessed_data = preprocess_data(data)

    # Make the prediction using the loaded model
    prediction = model.predict(preprocessed_data)

    # Return the prediction result as a JSON response
    response = {
        'prediction': prediction.tolist()
    }
    return jsonify(response)


def preprocess_data(data):
    preprocessed_data = data  # Placeholder, replace with actual preprocessing steps
    return preprocessed_data

if __name__ =="__main__":
    # app.run(debug=True,port=8000,host='0.0.0.0')
    app.run(host='0.0.0.0', port=os.getenv("PORT"),debug=True)