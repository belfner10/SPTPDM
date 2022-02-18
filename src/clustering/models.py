import numpy as np
from scipy.special import expit
import tensorflow as tf






if __name__ == '__main__':
    adj = np.load('adj.npy')
    adj = adj/np.argmax(adj)
    num_regions = 10
    d = 5
    w = np.random.random((num_regions,d))
    z = np.matmul(adj,w)
    reconstructed = expit(np.matmul(z,z.T))
    print(tf.nn.sigmoid_cross_entropy_with_logits(labels=adj,logits=reconstructed))
    # print(adj)
    # d = np.count_nonzero(adj,axis=0)
    # d = 1/np.sqrt(d)
    # D = np.diag(d)
    # print(D.dot(adj).dot(D))
    # print(d)
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    #
    # print(norm)