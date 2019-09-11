import os
import numpy as np
from HMM import hmm
from pre_process import parse_label
from util import save_obj
from config import *
from test import gesture_recognition_test


def training_single(file_path, file_names):
    """
    training hmm models for all 6 gestures respectively, each using one observation sequence
    :param file_path: path
    :param file_names: list of 6 files, one for each gesture
    :return: models, a dictionary containing 6 hmm models
    """
    models = {}
    for i, gesture in enumerate(gesture_names):
        print("\nTraining model for gesture \"{}\"...".format(gesture))
        label = parse_label(gesture)
        model = hmm(N[i], M) # initialize model
        for file_name in file_names:
            if parse_label(file_name) == label:
                data = np.load(file_path+file_name)
                _ = model.baum_welch(data['Z'])
                models[gesture] = model
                break

    return models

def training_multi(file_path, file_names):
    """
    training hmm models for all 6 gestures respectively, each using multiple observation sequence
    :param file_path: path
    :param file_names: list of 6 files, one for each gesture
    :return: models, a dictionary containing 6 hmm models
    """
    models = {}
    for i, gesture in enumerate(gesture_names):
        print("Training model for gesture \"{}\"...".format(gesture))
        label = parse_label(gesture)
        file_names_of_gesture = [name for name in file_names if name.startswith(gesture)]
        obs_seq_set = []
        model = hmm(N[i], M, use_scaling=True)
        for file_name in file_names_of_gesture:
            # if file_name in train_name_bank:
            data = np.load(file_path + file_name)
            obs_seq_set.append(data['Z'])

        model.baum_welch_multi(obs_seq_set)
        models[gesture] = model

    return models

# # train models using only one example each:


if __name__=="__main__":
    # train model
    file_names = [p for p in os.listdir(quant_file_path) if p.endswith('.npz')]
    models = training_multi(quant_file_path, file_names)

    # file_names_subset = ['beat3_31.npz', 'beat4_32.npz', 'circle14.npz', 'eight01.npz', 'inf16.npz', 'wave01.npz']
    # models = training_single(quant_file_path, file_names_subset)

    # save models
    save_obj(models, 'hmm_models.pkl')

    # compute training accuracy
    gesture_recognition_test(quant_file_path, file_names, hmm_models=models, quantized=True)


