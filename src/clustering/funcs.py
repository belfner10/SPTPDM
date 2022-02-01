import numpy as np
from sklearn.linear_model import LinearRegression


def l_method(x, y):
    out = np.zeros(len(x))
    for i in range(2, len(x) - 1):
        reg = LinearRegression().fit(x[:i], y[:i])
        score1 = reg.score(x[:i], y[:i])
        # print(score1)
        reg = LinearRegression().fit(x[i:], y[i:])
        score2 = reg.score(x[i:], y[i:])
        # print(score2)
        # print(score1+score2)
        out[i] = score1 + score2

    return np.argmax(out)