import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.interpolate import BarycentricInterpolator
import librosa

# ABSTRACT
class genre_classification:
    model = None
    standard_length = 1293

    def split_genre(self, tracks):
        genres = {}
        for t in tracks:
            track = tracks[t]
            a, sr = librosa.load(track.audio_path)
            g = genres.get(track.genre, [])
            g.append(self.extract_features(a, sr))
            genres[track.genre] = g
        return genres

    def split_data(self, tracks, ratio=.3):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        genres = self.split_genre(tracks)
        for g in genres:
            genre = genres[g]
            X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(genre, [g]*len(genre), test_size = ratio)
            print(y_train_t)
            print([g]*len(genre))
            print(y_train)
            if len(X_train) == 0:
                X_train = X_train_t 
                X_test = X_test_t
            else:
                X_train = np.concatenate((X_train, X_train_t))
                X_test = np.concatenate((X_test, X_test_t))
            y_train = np.concatenate((y_train, y_train_t))
            y_test = np.concatenate((y_test, y_test_t))
        # for t in tracks:
        #     track = tracks[t]
        #     a, sr = librosa.load(track.audio_path)
        #     X.append(self.extract_features(a, sr))
        #     y.append(track.genre)

        return X_train, X_test, y_train, y_test
    
    def extract_features(self, audio, sr):
        tmp_mfccs = (librosa.feature.mfcc(y=audio, sr=sr)).mean(axis=0)
        diff = self.standard_length - len(tmp_mfccs)
        mfccs = np.pad(tmp_mfccs, (0, diff if diff >= 0 else 0), mode='constant', constant_values=0)[:self.standard_length]
        
        # tmp_spec = (librosa.feature.spectral_centroid(y=audio, sr=sr)).mean(axis=0)
        # diff = self.standard_length - len(tmp_spec)
        # spec = np.pad(tmp_spec, (0, diff if diff >= 0 else 0), mode='constant', constant_values=0)[:1293]
        # features = np.stack([mfccs, spec], axis=1)
        return mfccs

    def interpolate(self, X):
        return BarycentricInterpolator(X)

    # abstract, implement in children
    def gen(self, n_estimators=100, random_state=42):
        pass

    def fit(self, X_train, y_train):
        if self.model == None:
            print("attempting to use model before generation. run gen() first.")
            return -1
        self.model.fit(X_train, y_train)

    def pred(self, X_test):
        if self.model == None:
            print("attempting to use model before generation. run gen() first.")
            return -1
        return self.model.predict(X_test)

    def eval(self, y_test, y_pred):
        accs = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        return accs, cr