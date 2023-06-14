import pickle
from flask import Flask, jsonify, request
import numpy as np
import parselmouth
from parselmouth.praat import call
import csv
from keras.models import load_model
import tensorflow as tf
from pydub import AudioSegment
import nolds
import base64
import os

app = Flask(__name__)

# Load the trained model
# model = load_model('content_model(CNN).pickle')
# model = pickle.load(open("content_model(CNN).pickle", "rb"))

# model_path = 'voicemodel.h5'
# loaded_model = tf.keras.models.load_model(model_path)

with open('content_model6xg.pkl', 'rb') as file:
    model = pickle.load(file)
@app.route('/upload', methods=['POST'])
def process_audio():
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
    # Perform any necessary preprocessing on the data
    preprocessed_data = data  # Placeholder, replace with actual preprocessing steps
    return preprocessed_data


if __name__ == '__main__':
    app.run(debug=True, port=8000,host='0.0.0.0')