
# visualization packages
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Standard data manipulation packages
import pandas as pd
import numpy as np

from sklearn.metrics import plot_confusion_matrix

def plot_cm(X_train, y_train, X_test, y_test, model):

    model.fit(X_train, y_train)
    y_hat_test = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(16,8))
    plt.rcParams.update({'font.size': 16})
    plot_confusion_matrix(model, 
                          X_test, 
                          y_test,
                          display_labels = ['Unvaxxed', 'Vaxxed'],
                          cmap = 'Blues',
                          ax=ax
                         )
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    pass

def result(data, title):
    
    fig, ax = plt.subplots(figsize=(7,5))
    bar1 = ax.bar([0, 1], data)
    plt.xticks(np.arange(2), ['Unvaccinated', 'Vaccinated']) 
    ax.yaxis.set_major_formatter('{x:.0%}')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.bar_label(bar1, fmt='%.2f')
    plt.tight_layout()
    plt.show()
    
    pass
