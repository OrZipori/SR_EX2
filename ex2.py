import librosa
import numpy as np 
import os

TEST_PATH = "./test_files/"
TRIAN_PATH = "./train_data/"
labels = [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5)]
sample_shape = 20 * 32

# dynamic programming implementation 
def DTW(src, trgt):
    dtw = np.zeros((src.shape[1], trgt.shape[1]))

    for i in range(src.shape[1]):
        for j in range(trgt.shape[1]):
            dtw[i][j] = np.inf
    
    dtw[i][j] = 0

    for i in range(1, src.shape[1]):
        for j in range(1, trgt.shape[1]):
            c = np.linalg.norm(src[i], trgt[j]) # Euclidean distance
            dtw[i][j] = c + np.min([dtw[i - 1][j],
                                    dtw[i][j - 1],
                                    dtw[i - 1][j - 1]])
    
    # get the dtw distance
    return dtw[src.shape[1], trgt.shape[1]]

# init knn table
def populate_knn_table():
    knn_table = []

    for label in labels:
        path = TRIAN_PATH + label[0]
        for f in os.listdir(path):
            # only wav files
            if (f.endswith(".wav")):
                y, sr = librosa.load(path + "/" + f, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                # reshape mfcc matrix to 620 feature vector
                knn_table.append([mfcc.reshape((sample_shape)), label[1]])
        
    return knn_table

def main():
    knn_table = populate_knn_table()

    

if __name__ == "__main__":
    main()


