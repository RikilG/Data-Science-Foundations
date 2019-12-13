# STATISTICS

## Assignment 1
### Normal Equations
solving Normal Equations normalized min-max
w = np.array([21.25266875, 13.52829146, -14.20940573)
MSE: 336.4022328613866,        RMSE: 18.34127129894181    time: 0.0146 sec

solving Normal Equations normalized normal
w = np.array([22.18057811, 2.77968805, -3.52160454])
MSE: 336.4022328613866,        RMSE: 18.34127129894181    time: 0.0163 sec

solving Normal Equations NOT normalized
w = np.array([673.50254732, 4.43085576, -12.16532254])
MSE: 336.4022328614068,        RMSE: 18.34127129894236    time: 0.0171 sec

### Standard Gradient Descent
stdgrad descent normalized min-max
alpha = 4e-6      epsilion=1e-9
w = np.array([ 21.252946, 13.527166, -14.208672])
MSE: 336.40226439250984,      RMSE: 18.34127215850934     time: 2.6 sec

stdgrad descent normalized normal
w = np.array([22.18057792,  2.77964608, -3.52156259])

stdgrad descent NOT normalized (Manual weights correction)
w = np.array([605.14287044, 4.21484986, -10.92459524])

### Stocastic GRAD Descent
stocastic grad descent normalized min-max
alpha = 1e-3
w = np.array([ 21.005, 13.968, -13.882])
MSE: 336.4424602330667,      RMSE: 18.34236790147517     time: 5.5 sec

stocastic grad descent normalized normal
alpha = 0.003
w = np.array([22.25783981, 2.78476251 -3.51462797])

### Regularization L1
### Regularization L2

## Assignment 2
### Normal Equations
### Regularization
Degree 1 :
    L1 :
        lambda      = -0.034492
        Train MSE   = 337.54769356941176
        Test MSE    = 337.2592732731232
        tr r2 = 18.372295596237716
        ts r2 = 18.36431757601515
        Train R2    = 0.026113402495594906
        Test R2     = 0.027366882336142928
    L2 :
        lambda      = -0.04311596
        Train MSE   = 338.60227
        Test MSE    = 338.13394
        Train R2    = 0.0230707
        Test R2     = 0.0248443

Degree 6 :
    L1 : 
        lambda      = -0.03736
        Train MSE   = 324.67932
        Test MSE    = 324.9320
        Train R2    = 0.06324
        Test R2     = 0.06291
    L2 :
        lambda      = -0.0373671
        Train MSE   = 326.58833
        Test MSE    = 326.8099
        Train R2    = 0.057733
        Test R2     = 0.057502

## Degree Analysis
Degree : 1      Error(MSE) : 337       Error(R2) : 0.026
Degree : 2      Error(MSE) :        Error(R2) : 
Degree : 3      Error(MSE) :        Error(R2) : 
Degree : 4      Error(MSE) :        Error(R2) : 
Degree : 5      Error(MSE) :        Error(R2) : 
Degree : 6      Error(MSE) :        Error(R2) : 
Degree : 7      Error(MSE) :        Error(R2) : 
Degree : 8      Error(MSE) :        Error(R2) : 
Degree : 9      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 1      Error(MSE) :        Error(R2) : 
Degree : 2      Error(MSE) :        Error(R2) : 