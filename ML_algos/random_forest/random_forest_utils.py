import mirdata
import librosa
import mir_eval
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import random

def rf_split_data(tracks, ratio=.3):
    X = []
    y = []
    for t in tracks:
        track = tracks[t]
        X.append(track.audio)
        y.append(track.genre)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    return train_test_split(X, y, test_size = 0.30)

    # train = {}
    # test = {}
    # for genre in tracks:
    #     subtracks = tracks[genre]
    #     len_subtracks = len(subtracks)
    #     train_tmp = []
    #     for _ in range(int(len_subtracks*(1-ratio))): 
    #         ind = random.randint(0,len(subtracks)-1)
    #         train_tmp.append(subtracks.pop(ind))``
            
    #     train[genre] = train_tmp
    #     test[genre] = subtracks

    # return train, test

def rf_gen(n_estimators=100, random_state=42):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

def rf_fit(rf, X_train, y_train):
    rf.fit(X_train, y_train)

def rf_pred(rf, X_test):
    return rf.predict(X_test)

def rf_eval(y_test, y_pred):
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)