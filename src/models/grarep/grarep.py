from scipy import sparse as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def norm_adj_sp(adj):
    counts = []
    for i in range(adj.shape[0]):
        _, cols = adj[i, :].nonzero()
        counts.append(len(cols))
    D = sp.diags(np.array(counts).astype('float32') ** -1)
    adj_normalized = D.dot(adj)
    return adj_normalized


def random_adj_sp(size):
    c = 0
    while True:
        b = np.random.randint(0, 2, (size, size))
        adj = np.tril(b) + np.tril(b).T
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_matrix(adj)
        if nx.is_connected(G):
            nx.draw(G)
            plt.show()
            print(c)
            return adj
        c += 1


if __name__ == '__main__':
    # sp.csr_matrix()
    size = 10
    adj = sp.csr_matrix(random_adj_sp(size))
    lambda_v = 3
    print(adj.toarray())
    print(norm_adj_sp(adj).toarray())
