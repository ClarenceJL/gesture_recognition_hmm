from pre_process import quantization_predict, load_obj
import numpy as np
from config import *
import os


def gesture_recognition_test(file_path, file_names, hmm_models, quantized=False, kmeans_model=[]):

    for i, file_name in enumerate(file_names):

        if quantized:
            data = np.load(file_path + file_name)  # data has already been quantized
            Z = data['Z']
        else:
            # load data
            data = np.loadtxt(file_path + file_name)
            # data quantization
            Z = quantization_predict(data[:, 1:7], kmeans_model)

        # calculate log-likelihood of each model:
        llhs = []
        for j, gesture in enumerate(gesture_names):
            model = hmm_models[gesture]
            llh = model.inference_forward(Z)[2]
            llhs.append(llh)

        # predict
        est = np.argmax(llhs)
        print("{} estimation: {}".format(file_name, gesture_names[est]))

        llhs = np.array(llhs)
        np.set_printoptions(precision=2)
        print(llhs)



if __name__=="__main__":
    file_names = [p for p in os.listdir(test_file_path) if p.endswith('.txt')]
    # load models
    hm = load_obj('hmm_models.pkl')
    km = load_obj('kmeans_model.pkl')
    # run gesture recognition:
    gesture_recognition_test(test_file_path, file_names, hm, quantized=False, kmeans_model=km)