from scipy import sparse as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

seed = 42
np.random.seed(seed)


def norm_adj_sp(adj):
    counts = []
    for i in range(adj.shape[0]):
        _, cols = adj[i, :].nonzero()
        counts.append(len(cols))
    D = sp.diags(np.array(counts).astype('float32') ** -1)
    adj_normalized = D.dot(adj)
    return adj_normalized


def random_adj_sp(size, frac):
    c = 0
    while True:
        b = np.random.randint(0, 2, (size, size))
        b = (np.random.random((size, size)) < frac) * 1
        adj = np.tril(b) + np.tril(b).T
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_matrix(adj)
        if nx.is_connected(G):
            # nx.draw(G)
            # plt.show()
            print(c)
            return adj
        c += 1


def get_components(ak, lambda_v, n_components=2, n_iter=20, rseed=None):
    ak = sp.coo_matrix(ak)
    ak.data = np.log(ak.data) - np.log(lambda_v / ak.shape[0])
    ak.col = np.array([cind for cind, val in zip(ak.col, ak.data) if val > 0], dtype=ak.col.dtype)
    ak.row = np.array([rind for rind, val in zip(ak.row, ak.data) if val > 0], dtype=ak.row.dtype)
    ak.data = np.array([val for val in ak.data if val > 0], dtype=ak.data.dtype)
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=rseed)
    comp = svd.fit_transform(ak)
    return comp


def get_k_components(adj, k, lambda_v, n_components=2, n_iter=20, rseed=None):
    norm = normalize(adj, 'l1', axis=1)
    ak = norm.copy()
    components = []
    for x in range(k):
        components.append(get_components(ak, lambda_v, n_components, n_iter, rseed))
        if x != k - 1:
            ak = ak.dot(norm)

    components = np.hstack(components)
    return components


if __name__ == '__main__':
    # sp.csr_matrix()
    size = 20
    adj = sp.csr_matrix(random_adj_sp(size, .35))
    lambda_v = 3
    print(get_k_components(adj, 2, lambda_v))
