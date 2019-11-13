from LinearModel.LinearModelStats import test
import DataUtils
import numpy as np
from tqdm import tqdm

f = None


# gradient function
def grad_f(w: np.array, x: np.array, y: np.array, lamda: float, reg_type: str) -> np.array:
    res = (y - f(w, x))
    w_grad = np.arange(w.shape[0])
    for i in range(w.shape[0]):
        temp  = -1*res*x[:, i].reshape(x.shape[0],1)
        if i==0: w_grad[i] = np.sum(temp)
        elif reg_type=="L2GD":   w_grad[i] = np.sum(temp) + lamda*w[i]
        elif reg_type=="L1GD":   w_grad[i] = np.sum(temp) + lamda*int(abs(w[i])/w[i])

    return w_grad


def grad_desc(x_train, y_train, error, alpha, epsilion, lamda, reg_type):
    w = np.ones(x_train.shape[1])
    iterations = 2
    prev_error = 0
    while iterations>0:
        # print(prev_error)
        w = w - (alpha*grad_f(w, x_train, y_train, lamda, reg_type))
        new_error = error(w, x_train, y_train)
        if abs(new_error-prev_error) < epsilion:
            break
        prev_error = new_error
        iterations -= 1
        if iterations == 1:
            iterations = 500
            # print(f"MSE: {new_error}, \tRMSE: {new_error**0.5}, \tWeights: {w}")
    return w


def run(data, function, error, alpha, epsilion=1e-9, reg_type="L2GD", lamdas=[]):
    global f
    f = function
    train, validate = DataUtils.data_split(data, split_at=0.8)\

    x_train, y_train = DataUtils.xy_split(train, target="ALTITUDE")
    x_val, y_val     = DataUtils.xy_split(validate, target="ALTITUDE")

    print(f"Starting L2 Regularized Gradient Descent\n with alpha={alpha}, epsilion={epsilion}\n")
    x_train.insert(0, "Const", np.ones(x_train.shape[0]))
    x_val.insert(0, "Const", np.ones(x_val.shape[0]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if len(lamdas) == 0:
        # lamdas = np.arange(0,1,0.1) # values b/w 0 and 1 stepped by 0.1
        lamdas = [i for i in range(-2000,2000,200)]

    val_err = list()
    train_err = list()
    w_list = list()
    for lamda in tqdm(lamdas):
        # print(f"\n>>>Lamda: {lamda}")
        w = grad_desc(x_train, y_train, error, alpha, epsilion, lamda, reg_type)
        w_list.append(w)
        val_err.append( test(w, x_val, y_val) )
        train_err.append( error(w, x_train, y_train) )

    return w_list, lamdas, val_err, train_err