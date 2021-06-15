import matplotlib.pyplot as plt
import numpy as np


def plot_cm(cm, title="Confusion Matrix", figsize=(6, 6), xticks=None, yticks=None,
            save_name=None, is_int: bool = False):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=24)
    plt.colorbar()
    length = len(cm[0])
    if xticks:
        plt.xticks(np.arange(length), xticks, rotation=45)
    if yticks:
        plt.yticks(np.arange(length), yticks)

    thresh = cm.max() / 2.
    for i in range(length):
        for j in range(length):
            if is_int:
                plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=12)
            else:
                plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.tight_layout()
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Pred label", fontsize=15)
    if save_name:
        plt.savefig(save_name)
    plt.show()
