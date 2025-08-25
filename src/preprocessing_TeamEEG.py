import os
import numpy as np
import importlib
import preprocessing_bci_2a_t  # Import the whole module first
import preprocessing_bci_2a_e

# Reload the module
importlib.reload(preprocessing_bci_2a_t)
importlib.reload(preprocessing_bci_2a_e)

from preprocessing_bci_2a_t import preprocess_bci_2a_t
from preprocessing_bci_2a_e import preprocess_bci_2a_e


os.getcwd()

data_folder = "../data"

bci_2a_folder = "/BCICIV_2a_gdf"
bci_2b_folder = "/BCICIV_2b_gdf"


file_list = [f for f in os.listdir(data_folder + bci_2a_folder) if f.endswith(".gdf")]

X_t = np.zeros(0)
y_t = np.zeros(0)
X_e = np.zeros(0)
y_e = np.zeros(0)

for f in file_list:

    print(f)

    if "T" in f:
        print("----------------------------------------------------")
        print(f"File name: {f}")
        print("----------------------------------------------------")

        X_f, y_f = preprocess_bci_2a_t(data_folder + bci_2a_folder + "/" + f)

        if X_t.size == 0:
            X_t = X_f
        else:
            X_t = np.concatenate((X_t, X_f), axis=0)

        if y_t.size == 0:
            y_t = y_f
        else:
            y_t = np.concatenate((y_t, y_f), axis=0)

        print("----------------------------------------------------")

    if "E" in f:
        # continue
        print("----------------------------------------------------")
        print(f"File name: {f}")
        print("----------------------------------------------------")

        X_f, y_f = preprocess_bci_2a_e(data_folder + bci_2a_folder + "/" + f)

        if X_e.size == 0:
            X_e = X_f
        else:
            X_e = np.concatenate((X_e, X_f), axis=0)

        if y_e.size == 0:
            y_e = y_f
        else:
            y_e = np.concatenate((y_e, y_f), axis=0)

        print("----------------------------------------------------")

np.save("X_bci_2a_training_data.npy", X_t)
np.save("y_bci_2a_training_data.npy", y_t)

np.save("X_bci_2a_evaluation_data_2.npy", X_e)
np.save("y_bci_2a_evaluation_data_2.npy", y_e)

