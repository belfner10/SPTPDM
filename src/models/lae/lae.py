import sys

import tensorflow as tf
import numpy as np
from scipy import sparse as sp

tf.compat.v1.disable_eager_execution()


# taken from linear graph autoencoders
def weight_variable_glorot(input_dim, output_dim, name="weights"):
    """
    Create a weight variable with Glorot&Bengio (AISTATS 2010) initialization
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# taken from linear graph autoencoders
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def actual_normalize(adj):
    adj_ = adj + np.identity(adj.shape[0])
    degrees = np.count_nonzero(adj_, axis=0) ** -0.5
    D = np.diag(degrees)
    adj_normalized = np.matmul(np.matmul(D, adj_), D)
    return adj_normalized


def normalize_sparse_adj(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    counts = np.array(adj_.sum(axis=0))[0] ** -0.5
    D = sp.diags(counts.astype('float32'))
    adj_normalized = D.dot(adj_).dot(D)
    return adj_normalized


def lae(adj_norm, adj, num_epochs, learning_rate=.001, components=10, threshold=.00001) -> np.ndarray:
    loss_vals = [sys.maxsize]
    adj_norm_tensor = tf.SparseTensor(*sparse_to_tuple(adj_norm.astype('float32')))
    adj_tensor = tf.sparse.to_dense(tf.SparseTensor(*sparse_to_tuple(adj.astype('float32'))))

    # necessary to register weight_tensor as a variable that can be changed during training
    with tf.compat.v1.variable_scope('vars'):
        weight_tensor = weight_variable_glorot(adj_norm.shape[0], components, name='weights')

    Z = tf.sparse.sparse_dense_matmul(adj_norm_tensor, weight_tensor)
    An = tf.math.sigmoid(tf.linalg.matmul(Z, tf.transpose(Z)))

    # TODO figure out if norm is necessary
    r = (adj != 0).astype(int).astype(adj.dtype)
    weight = float(adj.shape[0] * adj.shape[0] - r.sum()) / r.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * r.shape[0] - r.sum()) * 2)
    loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(adj_tensor, An, weight))

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    opt_op = opt.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    train_summary_writer = tf.compat.v1.summary.FileWriter('log_dir', sess.graph)
    with tf.compat.v1.name_scope('performance'):
        summary = tf.compat.v1.summary.scalar('loss', loss)

    for epoch in range(num_epochs):
        outs = sess.run([opt_op])
        loss_val = sess.run(loss)

        # for tensorboard
        sums = tf.compat.v1.summary.merge([summary])
        train_summary_writer.add_summary(sess.run(sums), epoch)

        if epoch % 10 == 0:
            print(f'Epoch: {str(epoch).rjust(len(str(num_epochs - 1)))}, Loss: {loss_val:0.4f} ')

        loss_vals.append(loss_val)

        if len(loss_vals) > 3 and abs(loss_vals[-3] - loss_vals[-2]) >= threshold and abs(loss_vals[-2] - loss_vals[-1]) < threshold:
            break


    weights = weight_tensor.eval(session=sess)
    return weights


def get_lae_comps(adj: np.ndarray, num_epochs: int = 500, learning_rate: float = .1, n_components: int = 10,
                  threshold: float = .00001) -> np.ndarray:
    if not isinstance(adj, sp.csr_matrix):
        adj = sp.csr_matrix(adj)
    adj_norm = normalize_sparse_adj(adj)
    return lae(adj_norm, adj, num_epochs=num_epochs,
               learning_rate=learning_rate,
               components=n_components,
               threshold=threshold)


def main():
    tf.random.set_seed(1)

    learning_rate = .1
    adj = sp.load_npz('adj_9122.npz')
    adj_norm = normalize_sparse_adj(adj)
    lae(adj_norm, adj, 1, learning_rate=learning_rate)


if __name__ == '__main__':
    main()
