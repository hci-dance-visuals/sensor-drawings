import os
from os.path import dirname, realpath
import h5py
import numpy as np
import cv2
from math import ceil, floor

name_filter = [".ipynb_checkpoints", "unprocessed", "scaled"]

dataset_directory = "sketch_feature_extractor/sketch_dataset/"
dataset_directory_2 = "sketch_feature_extractor/data/img"
test_dataset_directory = "../Soma_Draw/img_save/data/"

dancer_ids = {"Einav": 0,
            "Eleonora": 1
    }

def get_name(path, it):
    name = path
    for i in range(it):
        name = dirname(name)
    return os.path.splitext(os.path.basename(name))[0]

# Get image dataset from sketch_feature_extractor as np array
def loadDataset():
    exclude = set(["ipynb_checkpoints"])
    X_train = []
    # Load in the images
    filepaths = os.listdir(dataset_directory)
    filepaths[:] = [f for f in filepaths if "ipynb_checkpoints" not in f]
    for filepath in filepaths:
        # print(filepath)
        img = cv2.imread(dataset_directory+'{0}'.format(filepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_train.append(img)
    X_train = np.array(X_train)
    X_test = X_train[int(round(X_train.shape[0]*0.9)):X_train.shape[0],:,:]
    X_train = X_train.reshape([-1,112,112,1])/255.
    X_test = X_test.reshape([-1,112,112,1])/255.
    return X_train, X_test

def loadDatasetLabelled():
    exclude = set(["ipynb_checkpoints"])
    X_train = []
    Y_train = []
    seq_number = 0
    # Load in the images
    for path, subdirs, files, in os.walk(dataset_directory_2):
        subdirs.sort()
        for file in files:
            file_path = os.path.join(path,file)
            if "scaled" in file_path:
                id = str(int(ceil(len(X_train)/20)))
                label = get_name(file_path, 3) + str(int(ceil(len(X_train)/20))) + '-' + "%02d" % int(seq_number%20)
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X_train.append(img)
                Y_train.append(label)
                seq_number = seq_number + 1
    X_test = []
    Y_test = []
    seq_number = 0
    test_positions = [0]
    for path, subdirs, files, in os.walk(test_dataset_directory):
        subdirs.sort()
        for file in files:
            file_path = os.path.join(path,file)
            if "scaled" in file_path:
                id = str(int(ceil(len(X_test)/20)))
                label = get_name(file_path, 3) + str(int(ceil(len(X_test)/20))) + '-' + "%02d" % int(seq_number%20)
                try:
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    X_test.append(img)
                    Y_test.append(label)
                    seq_number = seq_number + 1
                except:
                    continue
    X_train = np.array(X_train)
    X_train = X_train.reshape([-1,112,112,1])/255.
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
#    X_test = X_train[int(round(X_train.shape[0]*0.9)):X_train.shape[0],:,:]
    X_test = X_test.reshape([-1,112,112,1])/255.
    return X_train, X_test, Y_train

if __name__ == "__main__":
    loadDatasetLabelled()
