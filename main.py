import LinearModel
import PolynomialModel
import DataUtils

import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop(columns="OSM_ID")    # drop unrequired feature

degree  = 1     # use 1 for linear model
# method  = "NE"
method  = "L2GD"

# add polynomial features if degree > 1
dataset = PolynomialModel.transform_dataset(dataset, degree)
# normalize data
dataset = DataUtils.normalize(dataset, type="min-max")
# split data
train, test = DataUtils.data_split(dataset, split_at=0.80)
# perform feature(x), target(y) split
x_train, y_train = DataUtils.xy_split(train)
x_test, y_test   = DataUtils.xy_split(test)
x_test.insert(0, "Const", np.ones(x_test.shape[0]))
x_train.insert(0, "Const", np.ones(x_train.shape[0]))
# print(f"x_train dimensions: {x_train.shape}")

st_time = time()
# w       = LinearModel.fit(x_train, y_train, alpha=3.8e-6, epsilion=1e-9, method=method)
# w_list, lamdas, val_errs, train_errs = LinearModel.reg_fit(train, alpha=4e-6, epsilion=1e-9, method=method)

print(f"Using polynomial of degree: {degree}")
# w       = PolynomialModel.fit(x_train, y_train, alpha=2.98e-6, epsilion=1e-4, method=method)
w_list, lamdas, val_errs, train_errs = PolynomialModel.reg_fit(train, alpha=7e-7, epsilion=1e-3, method=method, degree=degree)

test_errs = list()
for w in w_list:
    test_errs.append( LinearModel.test(w, x_test, y_test) )
plt.title(f'Error w.r.t lambda - degree: {degree}, method: {method}')
plt.ylabel('Error')
plt.xlabel('Lambda')
plt.plot(lamdas, val_errs, label="validation error")
# plt.plot(lamdas, train_errs, label="training error")
# plt.plot(lamdas, test_errs, label="testing error")
plt.legend()
plt.grid(True)
plt.show()
min_index = np.argmin(lamdas)
w       = w_list[min_index]
print("\nSelected Reg param: ", lamdas[min_index])

print(f"\nExecution Time : {time()-st_time}")
print(w)
print('Train Error(MSE):\t', LinearModel.error(w, x_train.values, y_train.values))
print('Test Error(MSE):\t', LinearModel.test(w, x_test, y_test))
print('Train Error(RMSE):\t', LinearModel.error(w, x_train.values, y_train.values)**0.5)
print('Test Error(RMSE):\t', LinearModel.test(w, x_test, y_test)**0.5)
print('Train Error(R2):\t', LinearModel.r2_error(w, x_train, y_train))
print('Test Error(R2):\t\t', LinearModel.r2_error(w, x_test, y_test))


# degree1: L1alpha = 4.5e-6       L2alpha = 6e-7
# degree2: L1alpha = 3.2e-6       L2alpha = 
# degree3: L1alpha = 2.98e-6      L2alpha = 
# degree4: L1alpha = 2.8e-6       L2alpha = 
# degree5: L1alpha = 2.7e-6       L2alpha = 
# degree6: L1alpha = 9.2e-7       L2alpha = 7e-7

# print distribution
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(dataset.iloc[:,0], dataset.iloc[:,1], dataset.iloc[:,2])
# plt.show()

# print correlation
# import seaborn as sns
# print(sns.heatmap(dataset.corr()))
# plt.show()
# exit()

# print heatmap
# import seaborn as sns
# print(sns.heatmap(dataset))
# plt.show()
# exit()

# using sklearn
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_train, y_train)
# print(model)
# print(model.intercept_, model.coef_)
# exit()