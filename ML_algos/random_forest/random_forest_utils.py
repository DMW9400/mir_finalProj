import mirdata
import librosa
import mir_eval
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random


# todo generalize
def rf_genre_split(tracks):
    genre_split = {}
    for tid in tracks:
        track = tracks[tid]
        tmp = genre_split.get(track.genre, []) #make dict?
        tmp.append(track)
        genre_split[track.genre] = tmp
    return genre_split

def rf_split_data(tracks, ratio=.8):
    train = {}
    test = {}
    for genre in tracks:
        subtracks = tracks[genre]
        len_subtracks = len(subtracks)
        train_tmp = []
        for _ in range(int(len_subtracks*(1-ratio))): 
            ind = random.randint(0,len(subtracks)-1)
            train_tmp.append(subtracks.pop(ind))
            
        train[genre] = train_tmp
        test[genre] = subtracks

    return train, test

def rf_gen(n_estimators=100, random_state=42):
    return RandomForestClassifier(n_estimators, random_state)

def rf_fit(rf, X_train, y_train):
    rf.fit(X_train, y_train)

def rf_pred(rf, X_test):
    return rf.predict(X_test)

def rf_eval(y_test, y_pred):
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)