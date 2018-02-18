import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
# https://qiita.com/Kodaira_/items/1a3b801c7a5a41c9ce49


def add_val(func_name, val):
    with open(func_name + '.dat', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([val])


def plot_val(func_name):
    df = pd.read_csv(func_name + '.dat')
    df.plot()
    plt.show()
    plt.close()
