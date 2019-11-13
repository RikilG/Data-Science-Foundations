import LinearModel
import PolynomialModel
import DataUtils

import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop(columns="OSM_ID")    # drop unrequired feature
# normalize data
dataset = DataUtils.normalize(dataset, type="min-max")
# split data
train, test = DataUtils.data_split(dataset, split_at=0.75)
# perform feature(x), target(y) split
x_train, y_train = DataUtils.xy_split(train, target="ALTITUDE")
x_test, y_test   = DataUtils.xy_split(test, target="ALTITUDE")
x_test.insert(0, "Const", np.ones(x_test.shape[0]))
print(f"x_train dimensions: {x_train.shape}")

st_time = time()
# w       = LinearModel.fit(x_train, y_train, alpha=4e-6, epsilion=1e-9, method="GD")
# w_list, lamdas, val_errs, train_errs = LinearModel.reg_fit(train, alpha=4e-6, epsilion=1e-9, method="L2GD")

degree  = 3
print(f"Using polynomial of degree: {degree}")
x_train = PolynomialModel.transform_dataset(x_train, degree)
x_test  = PolynomialModel.transform_dataset(x_test, degree)
w       = PolynomialModel.fit(x_train, y_train, alpha=3.1e-6, epsilion=1e-9, method="NE")
print(f"\nTime to find weights: {time()-st_time}")
print(w)
print(LinearModel.test(w, x_test, y_test))

# test_errs = list()
# for w in w_list:
#     test_errs.append(LinearModel.test(w, x_test, y_test))

# plt.title('Errors w.r.t lambda')
# plt.ylabel('MSE')
# plt.xlabel('lambda')
# plt.plot(lamdas, val_errs, label="validation error")
# plt.plot(lamdas, test_errs, label="testing error")
# plt.plot(lamdas, train_errs, label="training error")
# plt.legend()
# plt.grid(True)
# plt.show()