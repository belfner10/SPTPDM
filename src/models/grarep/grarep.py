import networkx as nx
import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import tqdm
from time import perf_counter

import multiprocessing as mp


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


def kernal(args):
    ak, lambda_v = args
    ak = sp.coo_matrix(ak)
    ak.data = np.log(ak.data) - np.log(lambda_v / ak.shape[0])
    ak.col = ak.col[ak.data > 0]
    ak.row = ak.row[ak.data > 0]
    ak.data = ak.data[ak.data > 0]
    return sp.csr_matrix(ak)


def create_aks(adj, k, lambda_v):
    norm = normalize(adj, 'l1', axis=1)
    a1 = norm.copy()
    aks = [(a1, lambda_v)]
    for x in range(k):
        aks.append((aks[x][0].dot(norm), 1))

    # p = mp.Pool(5)
    # results = p.imap_unordered(kernal, aks)
    # return list(results)
    for x in range(k):
        temp = aks[x][0]
        temp = sp.coo_matrix(temp)
        temp.data = np.log(temp.data) - np.log(lambda_v / temp.shape[0])
        temp.col = temp.col[temp.data > 0]
        temp.row = temp.row[temp.data > 0]
        temp.data = temp.data[temp.data > 0]
        aks[x] = sp.csr_matrix(temp)
        print(f'NNz: {aks[x].nnz}')
    return aks


def get_comps(aks, n_components=2):
    components = []
    t = tqdm.tqdm(total=sum([ak.nnz for ak in aks]))
    for ak in aks:
        # svd = TruncatedSVD(n_components=n_components, algorithm='arpack')
        svd = TruncatedSVD(n_components=n_components, n_iter=30)
        components.append(svd.fit_transform(ak))
        t.update(ak.nnz)
    components = np.hstack(components)
    return components


def get_grarep_comps(adj, k: int,
                     lambda_v: float = 1, n_components: int = 4,
                     n_iter: int = 50, random_seed: int = None) -> np.ndarray:
    norm = normalize(adj, 'l1')
    ak = norm.copy()
    components = []
    for x in tqdm.tqdm(range(k)):
        yk = normalize(ak.T, 'l1').T
        yk = sp.coo_matrix(yk)
        yk.data = np.log(yk.data) - np.log(lambda_v / yk.shape[0])
        yk.col = yk.col[yk.data > 0]
        yk.row = yk.row[yk.data > 0]
        yk.data = yk.data[yk.data > 0]
        yk = sp.csr_matrix(yk)
        svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_seed)
        components.append(svd.fit_transform(yk))
        if x != n_components - 1:
            ak = ak.dot(norm)
    components = np.hstack(components)
    components = normalize(components, 'l2')
    return components


# TODO Check if can be removed
def get_components(ak, lambda_v, n_components=2, n_iter=5, rseed=None):
    ak = sp.coo_matrix(ak)
    ak.data = np.log(ak.data) - np.log(lambda_v / ak.shape[0])
    ak.col = ak.col[ak.data > 0]
    ak.row = ak.row[ak.data > 0]
    ak.data = ak.data[ak.data > 0]
    ak = sp.csr_matrix(ak)

    print(f'NNz: {ak.nnz}')
    svd = TruncatedSVD(n_components=n_components, random_state=rseed, algorithm='arpack')
    comp = svd.fit_transform(ak)
    return comp


# TODO Check if can be removed
def get_k_components(adj, k, lambda_v, n_components=5, n_iter=20, rseed=None):
    norm = normalize(adj, 'l1', axis=1)
    ak = norm.copy()
    components = []
    for x in range(k):
        print(x)
        components.append(get_components(ak, lambda_v, n_components, n_iter, rseed))
        if x != k - 1:
            ak = ak.dot(norm)

    components = np.hstack(components)
    return components


if __name__ == '__main__':
    size = 6
    S = random_adj_sp(size, .35)
    print(S)
    D = np.diag([1 / float(x) for x in np.sum(S, axis=0)])
    adj = np.matmul(D,S)
    print(adj)

    print(normalize(S.T,'l1').T)


    # adj = sp.load_npz('data/adj_86105.npz')
    # lambda_v = 1
    # out = get_k_components(adj, 5, lambda_v, n_components=4)
    s = perf_counter()

    print(get_grarep_comps(S, k=2,lambda_v=1))
    print(perf_counter() - s)
    # np.1('out', out)
