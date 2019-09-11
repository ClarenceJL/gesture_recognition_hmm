"""
visualization
ref: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
"""
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle


def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def plot_conf(arr, col_names=[], row_names=[]):
    m, n = np.shape(arr)
    if len(col_names) != n:
        col_names = range(n)
    if len(row_names) != m:
        row_names = range(m)

    fig, ax = plt.subplots()
    cax = ax.matshow(arr, cmap=plt.get_cmap('viridis'))
    fig.colorbar(cax)
    plt.xticks(range(n), col_names)
    plt.yticks(range(m), row_names)

    # overlay text on heatmap
    x, y = np.meshgrid(np.arange(0, n), np.arange(0, m))
    for (x_val, y_val) in zip(x.flatten(), y.flatten()):
        c = "{:.2f}".format(arr[int(y_val), int(x_val)])
        ax.text(x_val, y_val, c, color='w', va='center', ha='center')

    plt.show()


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Code from: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
