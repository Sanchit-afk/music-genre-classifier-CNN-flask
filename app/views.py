from app import app
from flask import render_template
from flask import request, redirect
import os
import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
import functools as fp
import math
import matplotlib.pyplot as plt
from keras.models import load_model

@app.route("/")
def index():
    return render_template("/public/index.html")

app.config["original_path"] = "D:/Sem 6/flask/app"
app.config["SONG_UPLOADS"] = "D:/Sem 6/flask/app/app/static/music/uploads"
app.config["ALLOWED_MUSIC_EXTENSIONS"] = ["MP3", "WAV", "AU"]

def allowed_music(filename):

    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_MUSIC_EXTENSIONS"]:
        return True
    else:
        return False


def splitsongs(X, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = 33000
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return np.array(temp_X)


def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    # np.array([librosa.power_to_db(s, ref=np.max) for s in list(tsongs)])
    return np.array(list(tsongs))



def make_dataset_dl(args):
    # Convert to spectrograms and split into small windows
    signal, sr = librosa.load(args, sr=None)

    # Convert to dataset of spectograms/melspectograms
    signals = splitsongs(signal)

    # Convert to "spec" representation
    specs = to_melspectrogram(signals)

    return specs

genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

def get_genres(key, dict_genres):
    # Transforming data to help on transformation
    labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}

    return tmp_genre[key]

def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_genres(x, dict_genres), prob) for x, prob in votes.items()]



new_model = load_model('custom_cnn_2d.h5')


@app.route("/upload-song", methods = ["GET" , "POST"], defaults={})
def upload_song():
    
    if request.method == "POST":

        if request.files:

            song = request.files["document"]

            if not allowed_music(song.filename):
                print("That file extension is not allowed")
                return redirect(request.url)

            song.save(os.path.join(app.config["SONG_UPLOADS"] , song.filename))
            
            os.chdir(app.config["SONG_UPLOADS"])
            print(os.getcwd())
            
            specks = make_dataset_dl(song.filename)
            print(np.shape(specks))
            preds = new_model.predict(specks)
            votes = majority_voting(preds,genres)
            print(votes[0][0])
            genre = votes[0][0]
            genre = genre.upper()
            

            os.chdir(app.config["original_path"])
            print("song saved")
            
            
            
    
    return render_template("/public/result.html", data = genre)

@app.route("/about")
def about():
    return "All about Flask"

