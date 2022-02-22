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

def get_all_sub(cluster_id, Z, num_rows):
    if cluster_id < num_rows:
        return [cluster_id]
    else:
        return get_all_sub(int(Z[cluster_id - num_rows][0]), Z, num_rows) + get_all_sub(int(Z[cluster_id - num_rows][1]), Z, num_rows)

def create_clusters(Z, num_rows, num_clusters=2):
    cluster_seeds = [len(Z) + num_rows - 1]
    while len(cluster_seeds) < num_clusters:
        cid = max(cluster_seeds)
        cluster_seeds.remove(cid)
        cluster_seeds.append(int(Z[cid - num_rows][0]))
        cluster_seeds.append(int(Z[cid - num_rows][1]))

    ret = {}
    for x,cid in enumerate(cluster_seeds):
        for id in get_all_sub(cid, Z, num_rows):
            ret[id+1] = x
    return ret