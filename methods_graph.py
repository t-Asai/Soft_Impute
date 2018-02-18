import csv
import matplotlib.pyplot as plt
import pandas as pd
# https://qiita.com/Kodaira_/items/1a3b801c7a5a41c9ce49


def add_val(func_name, val='', flag=''):
    """
    グラフ描画用にデータを保存する
    """
    if flag == 'init':
        with open(func_name + '.dat', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([])
    else:
        with open(func_name + '.dat', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([val])


def plot_val(func_name):
    """
    グラフを描画する
    """
    df = pd.read_csv(func_name + '.dat', header=None)
    df.columns = ['']
    ax = df.plot()
    ax.set_xlabel('#step')
    ax.set_ylabel(func_name.replace('cal_', ''))
    plt.savefig('{}.eps'.format(func_name))
    plt.close()
