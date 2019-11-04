"""
This module implements standard gradient descent
"""
import numpy as np

f = None

# gradient function
def grad_f(w: np.array, x: np.array, y: np.array) -> int:
    res = (y - f(w, x))
    w_grad = np.arange(w.shape[0])
    for i in range(w.shape[0]):
        temp  = -1*res*x[:, i].reshape(x.shape[0],1)
        w_grad[i] = np.sum(temp)

    return w_grad

# main runner for standard gradient descent algorithm
def run(x_train, y_train, function, error, alpha, epsilion=1E-9):

    global f
    f = function
    print(f"Starting Gradient Descent with alpha={alpha}, epsilion={epsilion}\n")
    w       = np.ones(x_train.shape[1]+1)
    x_train.insert(0, "Const", np.ones(x_train.shape[0]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    iterations = 2
    prev_error = 0
    while iterations>0:
        # print(prev_error)
        w = w - (alpha*grad_f(w, x_train, y_train))
        new_error = error(w, x_train, y_train)
        if abs(new_error-prev_error) < epsilion:
            break
        prev_error = new_error
        iterations -= 1
        if iterations == 1:
            iterations = 500
            print(f"MSE: {new_error}, \tRMSE: {new_error**0.5}, \tWeights: {w}")

    return w