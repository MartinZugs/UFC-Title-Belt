import numpy as np

# path to the csv file
data_path = "manually-preprocessed_data-full-headers.csv"

# import data
data = np.genfromtxt(data_path, names=True, delimiter=",", dtype=np.float)

print(data)