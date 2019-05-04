import librosa
import numpy as np
import os
from scipy.spatial.distance import euclidean
from dtw import dtw
from scipy import stats

TEST_PATH = "./test_files/"
TRIAN_PATH = "./train_data/"
labels = [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5)]


# dynamic programming implementation
def DTW(src, trgt):
    dtw = np.full((src.shape[1]+1, trgt.shape[1]+1), np.inf, dtype=np.float)
    dtw[0,0] = 0

    for i in range(1, src.shape[1]+1):
        for j in range(1, trgt.shape[1]+1):
            c = euclidean(src[:,i-1], trgt[:,j-1]) # Euclidean distance
            #c = np.abs(src[i] - trgt[j])
            dtw[i][j] = c + np.min([dtw[i - 1][j],
                                    dtw[i][j - 1],
                                    dtw[i - 1][j - 1]])

    # get the dtw distance
    return dtw[-1, -1]


def matrix_euclidean_dist(src, trgt):
    '''
    :param src: mfcc feature matrix
    :param trgt: mfcc feature matrix
    :return: euclidean distance between two matrices, flatten
    '''
    sum = 0
    for i in range(src.shape[1]):
        sum+=euclidean(src[:,i], trgt[:,i])
    return sum/src.shape[1]

# init knn table
def populate_knn_table():
    knn_table = []

    for label in labels:
        path = TRIAN_PATH + label[0]
        for f in os.listdir(path):
            # only wav files
            if (f.endswith(".wav")):
                mfcc = load_file_feat(path + "/" + f)
                knn_table.append((mfcc, label[1]))

    return knn_table


def load_file_feat(path):
    '''
    :param path: files path
    :return: file's features
    '''
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return stats.zscore(mfcc,axis=1)


def do_knn(knn_table):
    test_set_classification = []
    same = 0
    test_files = [f for f in os.listdir(TEST_PATH) if ".wav" in f]

    for file in test_files:
        # Init result params
        dtw_dist = np.inf
        dtw_label = None
        euc_dist = np.inf
        euc_label = None
        test_ex_mfcc = load_file_feat(TEST_PATH+file)

        # Choose label with min. distance
        for ex_mfcc, ex_label in knn_table:
            curr_dtw_dist = DTW(ex_mfcc,test_ex_mfcc)
            curr_euc_dist = matrix_euclidean_dist(ex_mfcc, test_ex_mfcc)
            if curr_dtw_dist < dtw_dist:
                dtw_dist = curr_dtw_dist
                dtw_label = ex_label
            if curr_euc_dist < euc_dist:
                euc_dist = curr_euc_dist
                euc_label = ex_label
        test_set_classification.append((file, euc_label, dtw_label))
        if euc_label == dtw_label:
            same += 1
    print(f"Same is {same}/{len(test_set_classification)} - {same*100. / len(test_set_classification)}")

    return test_set_classification

def save_res(test_set_classification):
    output = open("output.txt", "w")
    for i, (file_name, euc_label, dtw_label) in enumerate(test_set_classification):
        output.write(f"{file_name} - {euc_label} - {dtw_label}")
        if i != len(test_set_classification) -1:
            output.write("\n")

def main():
    knn_table = populate_knn_table()
    test_res = do_knn(knn_table)
    save_res(test_res)

if __name__ == "__main__":
    main()
