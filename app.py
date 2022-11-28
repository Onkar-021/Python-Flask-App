from flask import Flask, render_template, request, jsonify
import pickle


import librosa
import numpy as np
#import pyaudio
import soundfile  # to read audio file

from gevent.pywsgi import WSGIServer
import os

app= Flask(__name__)
model = pickle.load(open("mlp_classifier.model", "rb"))


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result

@app.route('/')
def home():
    return render_template('index.html')
   
@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['file']
    
    
    features = extract_feature(file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    # show the result !
    Emotion= ("PREDITED  EMotion is : ", result)
    print(Emotion)
    
    

    classification = "Predicted Emotion: "+str(result)

    return render_template('index.html', prediction=classification)

port = os.getenv('VCAP_APP_PORT', '8080')

if __name__ == '__main__':
    #app.secret_key = os.uramdom(12)
    #app.run(port=port,debug=True, host='0.0.0.0')
    #app.run(debug=True)
    #port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)