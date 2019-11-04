import numpy as np

# test function
def test(w, x_test, y_test):
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    err = error(w, x_test, y_test)
    print(f"Testing Data:\n MSE: {err}, \tRMSE: {err**0.5}")
    # print(f"Weights: {w}")

# prediction function
def f(w: np.array, x: np.array) -> np.array:
    ans = x.dot(w)
    if ans.shape==(): return ans
    else: return x.dot(w).reshape(x.shape[0], 1)

# error fun: 1/2E(yn - w0-w1x1n-w2x2n)^2
def error(w: np.array, x: np.array, y: np.array) -> int:
    err = (y - f(w, x))**2
    return np.sum(err)/(err.shape[0])
