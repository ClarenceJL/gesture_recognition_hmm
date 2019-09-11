# LOOCV
import os
from config import *
from training import training_multi, training_single
from test import gesture_recognition_test

file_names = [p for p in os.listdir(quant_file_path) if p.endswith('.npz')]

cv_accuracy = 0
for test_file in file_names:
    print('test on file: {} ...'.format(test_file))
    train_files = file_names
    train_files.remove(test_file)
    models = training_multi(quant_file_path, file_names)
    test_result = gesture_recognition_test(quant_file_path, [test_file], hmm_models=models, quantized=True, stat_on=False)
    cv_accuracy = cv_accuracy + test_result

cv_accuracy = cv_accuracy / len(file_names)

print("Test accuracy for cross validation: {}".format(cv_accuracy))




