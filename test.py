from HMM import hmm
from pre_process import quantization_predict, parse_label, load_obj
from util import plot_conf
import numpy as np
from config import *
import os


def gesture_recognition_test(file_path, file_names, hmm_models, quantized=False, kmeans_model=[], stat_on=True):
    test_accuracy = 0
    conf_map = np.zeros([6, 6])
    num_instance_gesture = np.zeros([6])

    for i, file_name in enumerate(file_names):
        label = parse_label(file_name)
        num_instance_gesture[label] = num_instance_gesture[label] + 1

        if quantized:
            data = np.load(file_path + file_name)  # data has already been quantized
            Z = data['Z']
        else:
            # load data
            data = np.loadtxt(file_path + file_name)
            # data quantization
            # ts = data[:, 0]
            Z = quantization_predict(data[:, 1:7], kmeans_model)

        # calculate log-likelihood of each model:
        llhs = []
        for j, gesture in enumerate(gesture_names):
            model = hmm_models[gesture]
            llh = model.inference_forward(Z)[2]
            llhs.append(llh)

        # predict
        est = np.argmax(llhs)
        if est == label:
            test_accuracy = test_accuracy + 1
        conf_map[label, est] = conf_map[label, est] + 1

        if label == est:
            correct = "[CORRECT]"
        else:
            correct = "[WRONG]"
        print("{} true label: {}, estimation: {}   {}".format(file_name, gesture_names[label], gesture_names[est], correct))

        llhs = np.array(llhs)
        np.set_printoptions(precision=2)
        print(llhs)

    # display overall result
    test_accuracy = test_accuracy / len(file_names)

    if stat_on:
        print('The overall test accuracy is {}'.format(test_accuracy))
        num_instance_gesture = np.where(num_instance_gesture > 0, num_instance_gesture, 1)
        conf_map = conf_map / num_instance_gesture[:, np.newaxis]
        plot_conf(conf_map, col_names=gesture_names, row_names=gesture_names)

    return test_accuracy


if __name__=="__main__":
    file_names = [p for p in os.listdir(test_file_path) if p.endswith('.txt')]
    # load models
    hm = load_obj('hmm_models.pkl')
    km = load_obj('kmeans_model.pkl')
    # run gesture recognition:
    gesture_recognition_test(test_file_path, file_names, hm, quantized=False, kmeans_model=km)