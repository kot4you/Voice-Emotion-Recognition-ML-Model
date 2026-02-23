import multiprocessing as mp
import timeit
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import IPython.display as ipd
from IPython.display import Audio
import warnings

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# List all possible emotions present in all the datasets
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# The considered emotions that will be predicted
observed_emotions = ['neutral', 'disgust', 'happy',
                     'sad', 'angry', 'fearful', 'surprised']

# Parse the data from SAVEE dataset


def saveeEmotionParser(path):
    if path[9] == 'a':
        return "angry"
    elif path[9] == 'd':
        return "disgust"
    elif path[9] == 'f':
        return "fearful"
    elif path[9] == 'h':
        return "happy"
    elif path[9] == 'n':
        return "neutral"
    elif path[9] == 's':
        if path[10] == 'a':
            return "sad"
        return "surprised"

# Parse the data from CremaD dataset


def cremaEmotionParser(path):
    em = path.split("_")[2]
    if em == 'ANG':
        return "angry"
    elif em == 'DIS':
        return "disgust"
    elif em == 'FEA':
        return "fearful"
    elif em == 'HAP':
        return "happy"
    elif em == 'NEU':
        return "neutral"
    elif em == 'SAD':
        return "sad"

# Parse the data from TESS dataset


def TESSEmotionParser(path):
    em = path.split("_")[-1]
    if em == 'angry.wav':
        return "angry"
    elif em == 'disgust.wav':
        return "disgust"
    elif em == 'fear.wav':
        return "fearful"
    elif em == 'happy.wav':
        return "happy"
    elif em == 'neutral.wav':
        return "neutral"
    elif em == 'sad.wav':
        return "sad"
    elif em == 'ps.wav':
        return "surprised"


# List SAVEE
def list_SAVEE():
    emotionsSAVEE = []
    paths = []

    dirSAVEE = "/Users/juliusz/Library/Mobile Documents/com~apple~CloudDocs/Documents/Coding/IFE/Machine Learning/SAVEE"
    for r, d, f in os.walk(dirSAVEE):
        for file in f:
            filePath = os.path.join(r, file)
            emotionPath = filePath[filePath.index('SAVEE'):]

            emotion = saveeEmotionParser(emotionPath)

            try:
                if emotion not in observed_emotions:
                    continue

                emotionsSAVEE.append(emotion)
                paths.append(filePath)
            except:
                continue
    emotionDF = pd.DataFrame(emotionsSAVEE, columns=['Emotions'])
    pathDF = pd.DataFrame(paths, columns=['Path'])
    saveeDF = pd.concat([emotionDF, pathDF], axis=1)
    print("SAVEE loading Done!")
    return saveeDF

# List CremaD


def list_CremaD():
    emotionsCremaD = []
    paths = []

    dirCremaD = "/Users/juliusz/Library/Mobile Documents/com~apple~CloudDocs/Documents/Coding/IFE/Machine Learning/CremaD"
    for r, d, f in os.walk(dirCremaD):
        for file in f:
            filePath = os.path.join(r, file)
            emotionPath = filePath[filePath.index('CremaD'):]
            emotion = cremaEmotionParser(emotionPath)

            try:
                if emotion not in observed_emotions:
                    continue

                paths.append(filePath)
                emotionsCremaD.append(emotion)
            except:
                continue
    emotionDF = pd.DataFrame(emotionsCremaD, columns=['Emotions'])
    pathDF = pd.DataFrame(paths, columns=['Path'])
    cremaDF = pd.concat([emotionDF, pathDF], axis=1)
    print("CremaD loading Done!")
    return cremaDF

# List TESS


def list_TESS():
    emotionsTESS = []
    paths = []

    dirTESS = "/Users/juliusz/Library/Mobile Documents/com~apple~CloudDocs/Documents/Coding/IFE/Machine Learning/TESS"
    for r, d, f in os.walk(dirTESS):
        for file in f:
            filePath = os.path.join(r, file)
            emotionPath = filePath[filePath.index('TESS'):]
            emotion = TESSEmotionParser(emotionPath)

            try:
                if emotion not in observed_emotions:
                    continue

                paths.append(filePath)
                emotionsTESS.append(emotion)
            except:
                continue

    emotionDF = pd.DataFrame(emotionsTESS, columns=['Emotions'])
    pathDF = pd.DataFrame(paths, columns=['Path'])
    tessDF = pd.concat([emotionDF, pathDF], axis=1)
    print("TESS loading Done!")
    return tessDF


# Augumentation
# Add noise augmentation
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# Add time stretch augmentation


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

# Add time shift augmentation


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)

# Add pitch shift augmentation


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


ravdessDF = list_Ravdess()
saveeDF = list_SAVEE()
cremaDF = list_CremaD()
tessDF = list_TESS()

dataDF = pd.concat([ravdessDF, cremaDF, tessDF, saveeDF], axis=0)


# Feautre Extraction functions

# Zero crossing rate feature extraction


def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(
        data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

# root mean square error feature extraction


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(
        y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

# Mel coefficients feature extraction


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = librosa.feature.mfcc(
        y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc_result.T) if not flatten else np.ravel(mfcc_result.T)

# extraction of all the feautre types


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

# Get all the features and augmenations for a given path


def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = extract_features(data)
    audio = np.array(aud)

    noised_audio = noise(data)
    aud2 = extract_features(noised_audio)
    audio = np.vstack((audio, aud2))

    pitched_audio = pitch(data, sr)
    aud3 = extract_features(pitched_audio)
    audio = np.vstack((audio, aud3))

    pitched_audio1 = pitch(data, sr)
    pitched_noised_audio = noise(pitched_audio1)
    aud4 = extract_features(pitched_noised_audio)
    audio = np.vstack((audio, aud4))

    return audio

# make the final feature arrays


def makeFeatureArrays():
    start = timeit.default_timer()
    X, Y = [], []
    for path, emotion, index in tqdm(zip(dataDF.Path, dataDF.Emotions, range(dataDF.Path.shape[0]))):
        features = get_features(path)
        if index % 500 == 0:
            print(f'{index} audio has been processed')
        for i in features:
            X.append(i)
            Y.append(emotion)
    print('Feature arrays were made!')
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(f"The length of X is {len(X)}")
    print(f"The length of Y is {len(Y)}")
    print(f"The shape of dataDF is {dataDF.Path.shape}")
    return X, Y


# Save the data to csv
X, Y = makeFeatureArrays()
Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('features.csv', index=False)
