"""
Preprocess IMU data - filter and discretization
"""
import numpy as np
from config import *
from sklearn.cluster import KMeans
import os
from util import save_obj, load_obj

def parse_label(name):
    for i in range(6):
        if name.startswith(gesture_names[i]):
            return i

def quantization_predict(*args):
    imu_measurements = args[0]
    if len(args) == 1:
        # load model
        kmeans = load_obj('kmeans_model.pkl')
    else:
        kmeans = args[1]

    return kmeans.predict(imu_measurements)

def quantization_train(imu_measurements, k):
    """
    :param imu_measurements: [T, 6]
    :param k: number of possible measurements
    :return:
    """
    # run k-means
    model = KMeans(n_clusters=k)
    kmeans = model.fit(imu_measurements)
    labels = kmeans.labels_
    kmeans.labels_ = [] # save space
    save_obj(kmeans, 'kmeans_model.pkl')

    return labels


if __name__=="__main__":

    file_names = [p for p in os.listdir(file_path) if p.endswith('.txt')]
    X = np.zeros([0, 6])
    T = np.zeros([0])
    ranges = [0]
    print("loading data...")
    for file_name in file_names:
        # load data
        data = np.loadtxt(file_path+file_name)
        ts = data[:, 0]
        imu = data[:, 1:7]
        X = np.concatenate((X, imu), axis=0)
        T = np.concatenate((T, ts))
        ranges.append(ranges[-1]+np.shape(data)[0])

    # quantize the measurement
    print("clustering the measurement...")
    labels = quantization_train(X, M)
    print("save to new files...")
    if not os.path.exists(quant_file_path):
        os.makedirs(quant_file_path)

    for i, file_name in enumerate(file_names):
        # split data
        ts = T[ranges[i]:ranges[i+1]]
        q_imu = labels[ranges[i]:ranges[i+1]]
        np.savez(quant_file_path+file_name[:-4], ts=ts, Z=q_imu)

    # for new measurements:
    # Z = quantization_predict(X)













