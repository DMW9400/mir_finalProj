import mirdata
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def split_data(tracks):
    """
    Splits the provided dataset into training, validation, and test subsets based on genre,
    ensuring that each genre is split independently, and then merges them.

    Parameters
    ----------
    tracks : dict
        Dictionary of track objects where keys are track IDs and values are dataset.track objects.

    Returns
    -------
    tracks_train : list
        List of tracks belonging to the 'training' subset.
    tracks_validate : list
        List of tracks belonging to the 'validation' subset.
    tracks_test : list
        List of tracks belonging to the 'test' subset.
    """
    # Group tracks by genre
    genre_groups = {}
    for track in tracks.values():
        genre = getattr(track, "genre", "Unknown")
        if genre not in genre_groups:
            genre_groups[genre] = []
        genre_groups[genre].append(track)

    # Initialize final splits
    tracks_train = []
    tracks_validate = []
    tracks_test = []

    # Split each genre independently
    for genre, genre_tracks in genre_groups.items():
        # Shuffle tracks within the genre
        random.shuffle(genre_tracks)

        # Calculate split indices
        # Calculate split indices
        total_tracks = len(genre_tracks)
        train_end = int(total_tracks * 0.6)  # 70% for training
        validate_end = train_end + int(total_tracks * 0.1)  # 15% for validation

        # Split the genre tracks
        tracks_train += genre_tracks[:train_end]
        tracks_validate += genre_tracks[train_end:validate_end]
        tracks_test += genre_tracks[validate_end:]

    return tracks_train, tracks_validate, tracks_test

    


def compute_mfccs(y, sr, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Compute mfccs for an audio file using librosa, removing the 0th MFCC coefficient.
    
    Parameters
    ----------
    y : np.array
        Mono audio signal
    sr : int
        Audio sample rate
    n_fft : int
        Number of points for computing the fft
    hop_length : int
        Number of samples to advance between frames
    n_mels : int
        Number of mel frequency bands to use
    n_mfcc : int
        Number of mfcc's to compute
    
    Returns
    -------
    mfccs: np.array (t, n_mfcc - 1)
        Matrix of mfccs

    """
    # YOUR CODE HERE
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    mfccs = mfccs[1:, :]

    mfccs = mfccs.T

    return mfccs

def get_stats(features):
    """
    Compute summary statistics (mean and standard deviation) over a matrix of MFCCs.
    Make sure the statitics are computed across time (i.e. over all examples, 
    compute the mean of each feature).

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features

    Returns
    -------
    features_mean: np.array (n_features)
                   The mean of the features
    features_std: np.array (n_features)
                   The standard deviation of the features

    """
    # Compute the mean across the examples for each feature 
    features_mean = np.mean(features, axis=0)
    
    # Compute the standard deviation across the examples for each feature 
    features_std = np.std(features, axis=0)
    
    return features_mean, features_std
    # Hint: use numpy mean and std functions, and watch out for the axis.
    # YOUR CODE HERE


def normalize(features, features_mean, features_std):
    """
    Normalize (standardize) a set of features using the given mean and standard deviation.

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features
    features_mean: np.array (n_features)
              The mean of the features
    features_std: np.array (n_features)
              The standard deviation of the features

    Returns
    -------
    features_norm: np.array (n_examples, n_features)
                   Standardized features

    """
    features_norm = (features - features_mean) / features_std
    
    return features_norm

    # YOUR CODE HERE


import audioread
import numpy as np
import scipy.signal
from scipy.io.wavfile import write

# def get_features_and_labels(track_list, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
#     """
#     Extracts features (mean and std MFCC values) and labels for all tracks using audioread.

#     Parameters
#     ----------
#     track_list : list
#         List of dataset.track objects, where each object contains `audio_path` and `genre`.
#     n_fft : int
#         Number of points for computing the FFT.
#     hop_length : int
#         Number of samples to advance between frames.
#     n_mels : int
#         Number of mel frequency bands to use.
#     n_mfcc : int
#         Number of MFCCs to compute.

#     Returns
#     -------
#     feature_matrix : np.array
#         Matrix of features for each track, shape (len(track_list), 2 * (n_mfcc - 1)).
#     label_array : np.array
#         Array of labels for each track.
#     """
#     features = []
#     labels = []

#     for track in track_list:
#         try:
#             # Load audio with audioread
#             with audioread.audio_open(track.audio_path) as f:
#                 sr = f.samplerate  # Sampling rate
#                 audio = np.frombuffer(b"".join([buf for buf in f]), dtype=np.int16)
#                 audio = audio.astype(np.float32) / np.max(np.abs(audio))  # Normalize to [-1, 1]

#             # Ensure mono signal
#             if len(audio.shape) > 1:
#                 audio = np.mean(audio, axis=1)

#             # Compute MFCCs
#             mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#             mfccs = mfccs[1:, :]  # Remove the 0th coefficient

#             # Compute mean and std
#             feature_mean = np.mean(mfccs, axis=1)
#             feature_std = np.std(mfccs, axis=1)

#             # Concatenate features
#             feature_vector = np.concatenate((feature_mean, feature_std))

#             # Append feature vector and label
#             features.append(feature_vector)
#             labels.append(track.genre)

#         except Exception as e:
#             print(f"Skipping {track.audio_path} due to error: {e}")
#             continue

#     # Convert lists to numpy arrays
#     feature_matrix = np.array(features)
#     label_array = np.array(labels)

#     return feature_matrix, label_array

def compute_chroma_stft(y, sr, n_fft=2048, hop_length=512):
    """Compute chroma short-time Fourier transform."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return chroma

def compute_rms_mean(y, sr, frame_length=2048, hop_length=512):
    """Compute mean root mean square (RMS) energy."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    return rms

def compute_spectral_centroid(y, sr, n_fft=2048, hop_length=512):
    """Compute spectral centroid."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return centroid

def compute_spectral_bandwidth(y, sr, n_fft=2048, hop_length=512):
    """Compute spectral bandwidth."""
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return bandwidth

def compute_rolloff(y, sr, n_fft=2048, hop_length=512, roll_percent=0.85):
    """Compute spectral roll-off."""
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent)
    return rolloff

def compute_zero_crossing_rate(y, hop_length=512):
    """Compute zero-crossing rate."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    return zcr

def get_features_and_labels(track_list, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Extracts features (MFCCs, chroma, RMS, spectral centroid, bandwidth, roll-off, ZCR) and labels for all tracks.

    Parameters
    ----------
    track_list : list
        List of dataset.track objects, where each object contains `audio_path` and `genre`.
    n_fft : int
        Number of points for computing the FFT.
    hop_length : int
        Number of samples to advance between frames.
    n_mels : int
        Number of mel frequency bands to use.
    n_mfcc : int
        Number of MFCCs to compute.

    Returns
    -------
    feature_matrix : np.array
        Matrix of features for each track, shape (len(track_list), feature_dimension).
    label_array : np.array
        Array of labels for each track.
    """
    features = []
    labels = []

    for track in track_list:
        try:
            # Load audio
            y, sr = librosa.load(track.audio_path, sr=None)

            # Compute features
            mfccs = compute_mfccs(y, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
            chroma = compute_chroma_stft(y, sr, n_fft=n_fft, hop_length=hop_length)
            rms = compute_rms_mean(y, sr)
            centroid = compute_spectral_centroid(y, sr, n_fft=n_fft, hop_length=hop_length)
            bandwidth = compute_spectral_bandwidth(y, sr, n_fft=n_fft, hop_length=hop_length)
            rolloff = compute_rolloff(y, sr, n_fft=n_fft, hop_length=hop_length)
            zcr = compute_zero_crossing_rate(y, hop_length=hop_length)

            # Summarize features using mean and variance
            mfccs_mean = np.mean(mfccs, axis=0)
            mfccs_var = np.var(mfccs, axis=0)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_var = np.var(chroma, axis=1)
            rms_mean = np.mean(rms)
            rms_var = np.var(rms)
            centroid_mean = np.mean(centroid)
            centroid_var = np.var(centroid)
            bandwidth_mean = np.mean(bandwidth)
            bandwidth_var = np.var(bandwidth)
            rolloff_mean = np.mean(rolloff)
            rolloff_var = np.var(rolloff)
            zcr_mean = np.mean(zcr)
            zcr_var = np.var(zcr)

            # Concatenate features
            feature_vector = np.concatenate((
                mfccs_mean, mfccs_var, chroma_mean, chroma_var,
                [rms_mean, rms_var, centroid_mean, centroid_var, 
                 bandwidth_mean, bandwidth_var, rolloff_mean, rolloff_var, 
                 zcr_mean, zcr_var]
            ))

            # Append feature vector and label
            features.append(feature_vector)
            labels.append(track.genre)

        except Exception as e:
            print(f"Skipping {track.audio_path} due to error: {e}")
            continue

    # Convert lists to numpy arrays
    feature_matrix = np.array(features)
    label_array = np.array(labels)

    return feature_matrix, label_array

def fit_knn(train_features, train_labels, validation_features, validation_labels, ks=[1, 5, 10, 50]):
    """
    Fit a k-nearest neighbor classifier and choose the k which maximizes the
    *f-measure* on the validation set.
    
    Plot the f-measure on the validation set as a function of k.

    Parameters
    ----------
    train_features : np.array (n_train_examples, n_features)
        training feature matrix
    train_labels : np.array (n_train_examples)
        training label array
    validation_features : np.array (n_validation_examples, n_features)
        validation feature matrix
    validation_labels : np.array (n_validation_examples)
        validation label array
    ks: list of int
        k values to evaluate using the validation set

    Returns
    -------
    knn_clf : scikit learn classifier
        Trained k-nearest neighbor classifier with the best k
    best_k : int
        The k which gave the best performance
    """
    
    # Hint: for simplicity you can search over k = 1, 5, 10, 50. 
    # Use KNeighborsClassifier from sklearn.
    # YOUR CODE HERE
    
    
    f1_scores = []  # Initialize an empty list to store f1 scores
    best_f1 = 0
    best_k = None
    knn_clf = None
    
    for k in ks:
        # Initialize and fit the k-nearest neighbors classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)
        
        # Predict labels for the validation set
        val_predictions = knn.predict(validation_features)
        
        # Calculate the f1 score on the validation set
        f1 = f1_score(validation_labels, val_predictions, average='weighted')
        f1_scores.append(f1)  # Append the f1 score to the list
        
        # Update the best k and classifier if the current f1 score is better
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            knn_clf = knn
    # Plot the f1 score as a function of k
    plt.figure(figsize=(8, 6))
    plt.plot(ks, f1_scores, marker='o')
    plt.title("F1-Measure vs K (Validation Set)")
    plt.xlabel("K")
    plt.ylabel("F1-Measure")
    plt.grid(True)
    plt.show()
    
    return knn_clf, best_k
    
    

from sklearn.model_selection import cross_val_score
def fit_decision_tree(train_features, train_labels, validation_features, validation_labels, depths=[3, 5, 10, 20, None]):
    """
    Fit a Decision Tree Classifier and choose the depth which maximizes the
    *f-measure* on the validation set using cross-validation.

    Parameters
    ----------
    train_features : np.array (n_train_examples, n_features)
        Training feature matrix.
    train_labels : np.array (n_train_examples)
        Training label array.
    validation_features : np.array (n_validation_examples, n_features)
        Validation feature matrix.
    validation_labels : np.array (n_validation_examples)
        Validation label array.
    depths : list of int or None
        Maximum depth values to evaluate using the validation set.

    Returns
    -------
    dt_clf : scikit-learn classifier
        Trained Decision Tree classifier with the best max_depth.
    best_depth : int or None
        The max_depth which gave the best performance.
    """

    f1_scores = []  # To store F1 scores
    best_f1 = 0
    best_depth = None
    dt_clf = None

    for depth in depths:
        # Initialize the Decision Tree with additional regularization
        dt = DecisionTreeClassifier(criterion='gini',
            splitter='best', 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0.0, 
            max_features=None, 
            random_state=None, 
            max_leaf_nodes=None, 
            min_impurity_decrease=0.0, 
            class_weight=None, 
            ccp_alpha=0.0, 
            monotonic_cst=None)
        

        # Perform cross-validation on the training set
        cv_f1_scores = cross_val_score(dt, train_features, train_labels, cv=5, scoring='f1_weighted')

        # Compute the mean F1 score from cross-validation
        mean_f1 = np.mean(cv_f1_scores)
        f1_scores.append(mean_f1)

        # Check if this depth gives the best F1 score
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_depth = depth
            dt_clf = dt  # Save the classifier with the best depth

    # Train the best model on the full training set
    dt_clf.fit(train_features, train_labels)

    # Plot the F1 scores as a function of max_depth
    plt.figure(figsize=(8, 6))
    plt.plot([str(d) for d in depths], f1_scores, marker='o')
    plt.title("F1-Measure vs Max Depth (Validation Set with Cross-Validation)")
    plt.xlabel("Max Depth")
    plt.ylabel("F1-Measure")
    plt.grid(True)
    plt.show()

    return dt_clf, best_depth
