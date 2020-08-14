# Imports
#General
import numpy as np
# Machine Learning
from sklearn.preprocessing import LabelEncoder
import joblib

# Random Seed
from numpy.random import seed
seed(1)

# Audio
import librosa.display, librosa

# Parameters
# Signal Processing Parameters
fs = 44100         # Sampling Frequency
n_mels = 128       # Number of Mel bands
n_mfcc = 20       # Number of MFCCs
mmeanPath = 'SVMClfParams\TraningDatasetMean\DatasetMean.csv'
mstdPath = 'SVMClfParams\TraningDatasetStd\DatasetStd.csv'
datasetClassesPath = '.\SVMClfParams\TrainigDatasetClasses\DatasetClasses.npy'
clfPath = './SVMClfParams\TrainedSVM/trainedSVM.joblib'

class Predictor(object):
    mmean = None
    mstd = None
    encoder = None
    svclassifier = None
    def __init__(self):
        # load mmean
        self.mmean = np.loadtxt(mmeanPath, delimiter=',')
        # load mstd
        self.mstd = np.loadtxt(mstdPath, delimiter=',')
        # load Classes
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(datasetClassesPath)
        # Load the Svm Clf
        self.svclassifier = joblib.load(clfPath)


    def get_features(self, y, sr=fs):
        S = librosa.feature.melspectrogram(y, sr=fs, n_mels=n_mels)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(mfcc, x_axis='time')
        # plt.colorbar()
        # plt.title('MFCC')
        # plt.tight_layout()
        feature_vector = np.mean(mfcc, 1)
        # feature_vector = (feature_vector-np.mean(feature_vector))/np.std(feature_vector)
        return feature_vector

    def get_scaled_feature_vector(self, OriginalSample):
        feature_vector = []
        try:
            y = OriginalSample/1
            sr = fs
            y /= y.max()  # Normalize
            if len(y) < 2:
                print("Error Sample length is too short")
            feat = self.get_features(y, sr)
            feature_vector.append(feat)
        except Exception as e:
            print("Error Sample length is too short. Error: %s" % (e))

        # print("Calculated %d feature vectors" % len(feature_vector))

        mfeature_vector = (feature_vector - self.mmean) / self.mstd
        scaled_feature_vector = mfeature_vector
        # print('Feature vector::', scaled_feature_vector)
        # print("Feature vector shape:", scaled_feature_vector.shape)
        return scaled_feature_vector

    def getPrediction(self, OriginalSample):
        scaled_feature_vector = self.get_scaled_feature_vector(OriginalSample)

        prediction = self.svclassifier.predict(scaled_feature_vector)
        return self.encoder.classes_[prediction]