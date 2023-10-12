from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import MusicFile
from .serializers import MusicFileSerializer
import numpy as np
import librosa
import tensorflow as tf

genres = {
    'Metal': 0, 'Disco': 1, 'Classical': 2, 'HipHop': 3, 'Jazz': 4, 
    'Country': 5, 'Pop': 6, 'Blues': 7, 'Reggae': 8, 'Rock': 9
}

class MusicFileList(generics.ListCreateAPIView):
    queryset = MusicFile.objects.all()
    serializer_class = MusicFileSerializer

class MusicFileDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = MusicFile.objects.all()
    serializer_class = MusicFileSerializer

def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_genres(x, dict_genres), prob) for x, prob in votes.items()]

def get_genres(key, dict_genres):
    # Transforming data to help on transformation
    labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}
    return tmp_genre[key]


class MusicGenreDetect(APIView):
    def post(self, request, format=None):
        audio_file = request.FILES['audio_file']
        y, sr = librosa.load(audio_file)

        X = splitsongs(y)
        # X = np.abs(librosa.stft(y))

        # X_test = librosa.feature.mfcc(S=X, n_mfcc=20)
        X_test = to_melspectrogram(X)

        # X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1], 1)
        model = tf.keras.models.load_model('model.h5')
        prediction = model.predict(X_test)
        # genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # result = genre[np.argmax(prediction)]
        # return Response({"genre": result})
        votes = majority_voting(prediction, genres)
        # print("{} is a {} song".format(self.args.song, votes[0][0]))
        print("{} is a {} song".format(audio_file, votes[0][0]))
        
        print("Most likely genres are: {}".format(votes[:3]))
        return Response({"Result": votes[0][0],"Possible Genres": votes[:3]})


def get_features(y, sr, n_fft = 1024, hop_length = 512):
    # Features to concatenate in the final dictionary
    features = {'centroid': None, 'roloff': None, 'flux': None, 'rmse': None,
                'zcr': None, 'contrast': None, 'bandwidth': None, 'flatness': None}
    
    # Count silence
    if 0 < len(y):
        y_sound, _ = librosa.effects.trim(y, frame_length=n_fft, hop_length=hop_length)
    features['sample_silence'] = len(y) - len(y_sound)

    # Using librosa to calculate the features
    features['centroid'] = librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['roloff'] = librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['rmse'] = librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['flux'] = librosa.onset.onset_strength(y=y, sr=sr).ravel()
    features['contrast'] = librosa.feature.spectral_contrast(y, sr=sr).ravel()
    features['bandwidth'] = librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['flatness'] = librosa.feature.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel()
    
    # MFCC treatment
    mfcc = librosa.feature.mfcc(y, n_fft = n_fft, hop_length = hop_length, n_mfcc=13)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()
        
    # Get statistics from the vectors
    def get_moments(descriptors):
        result = {}
        for k, v in descriptors.items():
            result['{}_max'.format(k)] = np.max(v)
            result['{}_min'.format(k)] = np.min(v)
            result['{}_mean'.format(k)] = np.mean(v)
            result['{}_std'.format(k)] = np.std(v)
            result['{}_kurtosis'.format(k)] = kurtosis(v)
            result['{}_skew'.format(k)] = skew(v)
        return result
    
    dict_agg_features = get_moments(features)
    dict_agg_features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]
    
    return dict_agg_features


"""
@description: Method to split a song into multiple songs using overlapping windows
"""
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

"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(y=x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    # np.array([librosa.power_to_db(s, ref=np.max) for s in list(tsongs)])
    return np.array(list(tsongs))
  

def make_dataset_ml(args):
    signal, sr = librosa.load(args.song)
    
    # Append the result to the data structure
    features = get_features(signal, sr)
    song = pd.DataFrame([features])
    return song


def make_dataset_dl(args):
    # Convert to spectrograms and split into small windows
    signal, sr = librosa.load(args.song)

    # Convert to dataset of spectograms/melspectograms
    signals = splitsongs(signal)

    # Convert to "spec" representation
    specs = to_melspectrogram(signals)

    return specs
    

