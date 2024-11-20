import mirdata
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score 
from sklearn.neighbors import KNeighborsClassifier

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
        total_tracks = len(genre_tracks)
        train_end = int(total_tracks * 0.6)  # 60% for training
        validate_end = int(total_tracks * 0.9)  # Next 30% for validation

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


def get_features_and_labels(track_list, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Our features are going to be the `mean` and `std` MFCC values of a track concatenated 
    into a single vector of size `2*n_mfcss`. 

    Create a function `get_features_and_labels()` such that extracts the features 
    and labels for all tracks in the dataset, such that for each audio file it obtains a 
    single feature vector. This function should do the following:

    For each track in the collection (e.g. training split),
        1. Compute the MFCCs of the input audio, and remove the first (0th) coeficient.
        2. Compute the summary statistics of the MFCCs over time:
            1. Find the mean and standard deviation for each MFCC feature (2 values for each)
            2. Stack these statistics into single 1-d vector of lenght ( 2 * (n_mfccs - 1) )
        3. Get the labels. The label of a track can be accessed by calling `track.instrument_id`.
    Return the labels and features as `np.arrays`.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset
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
    feature_matrix: np.array (len(track_list), 2*(n_mfcc - 1))
        The features for each track, stacked into a matrix.
    label_array: np.array (len(track_list))
        The label for each track, represented as integers
    """

    features = []
    labels = []
    
    # Iterate through each track in the track list
    for track in track_list:
        # Load the audio file using the audio path from the track object
        y, sr = librosa.load(track.audio_path, sr=None)

        # Compute MFCCs (remove the 0th coefficient)
        #mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs = compute_mfccs(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        mfccs = mfccs[1:, :]  # Remove the 0th coefficient

        # Compute mean and standard deviation for each MFCC coefficient
        #mfcc_mean = np.mean(mfccs, axis=1)
        #mfcc_std = np.std(mfccs, axis=1)
        feature_mean, feature_std = get_stats(mfccs)

        # Concatenate the mean and standard deviation into a single feature vector
        feature_vector = np.concatenate((feature_mean, feature_std))

        # Append the feature vector to the list of features
        features.append(feature_vector)

        # Get the label (instrument_id) for the current track
        labels.append(track.instrument_id)
    
    # Convert lists to numpy arrays
    feature_matrix = np.array(features)
    label_array = np.array(labels)

    return feature_matrix, label_array
    # Hint: re-use functions from previous parts (e.g. compute_mfcss and get_stats)
    # YOUR CODE HERE



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
    
    

