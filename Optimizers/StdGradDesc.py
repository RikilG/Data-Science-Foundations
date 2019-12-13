"""
This module implements standard gradient descent
"""
import numpy as np
from tqdm import tqdm

f = None

# gradient function
def grad_f(w: np.array, x: np.array, y: np.array) -> int:
    res = (y - f(w, x))
    w_grad = -1*(res.transpose().dot(x)).flatten()

    return w_grad

# main runner for standard gradient descent algorithm
def run(x_train, y_train, function, error, alpha, epsilion=1E-9):

    global f
    f = function
    print(f"Starting Gradient Descent with alpha={alpha}, epsilion={epsilion}\n")
    w       = np.ones(x_train.shape[1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    prev_error = 0
    prev_w = w
    while True:
        for iterations in tqdm(range(500)):
            grad = grad_f(w, x_train, y_train) # alpha multiplied inside
            # prev_w = w
            w = w - alpha*grad
            # if w.dot(prev_w) > 0.01:
            #     print(w.dot(prev_w))
            new_error = error(w, x_train, y_train)
            if new_error > 1e7:
                print("\nOVERSHOOT!!"*3)
                print(new_error)
                return w
            if abs(new_error-prev_error) < epsilion:
                return w
            prev_error = new_error
            
        print(f"MSE: {new_error}, \tRMSE: {new_error**0.5}, \tWeights: {w}\n")

    return w