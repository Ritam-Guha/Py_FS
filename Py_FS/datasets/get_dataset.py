import pandas as pd
import numpy as np
import os, glob, sys
from tabulate import tabulate

class Data():
    def __init__(self):
        self.data = None
        self.target = None

orig_filepaths = list(glob.iglob('Py_FS/datasets/database/*/*', recursive=True))
list_datasets = [os.path.splitext(os.path.basename(filename))[0] for filename in orig_filepaths]
list_subtypes = [os.path.basename(os.path.dirname(filename)) for filename in orig_filepaths]

def get_dataset(dataset_name):
    if dataset_name not in list_datasets:
        print(f"[!Error] Py_FS currently does not have {dataset_name} in its database....")
        display = input("Enter 1 to see the available datasets: ") or 0
        if display:
            display_datasets()

    else:
        data = Data()
        index = list_datasets.index(dataset_name)
        file_name = orig_filepaths[index]
        df = pd.read_csv(file_name, header=None)
        data.data = np.array(df.iloc[:, 0:-1])
        data.target = np.array(df.iloc[:, -1])

        print(data.data.shape)
        print(data.target.shape)


def display_datasets():
    print("\n=========== Available Datasets ===========\n")
    table_list = []

    for i, dataset in enumerate(list_datasets):
        table_list.append([i+1, list_datasets[i], list_subtypes[i]])

    print(tabulate(table_list, headers=["Index", "Dataset", "Subtype"]))

if __name__ == '__main__':
    get_dataset('Vow')
