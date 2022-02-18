import tensorflow as tf
import numpy as np
from scipy import sparse as sp
from time import perf_counter

# tf.compat.v1.disable_eager_execution()


#
# class MyMatMulLayer(tf.keras.layers.Layer):
#     def __init__(self, num_outputs):
#         super(MyMatMulLayer, self).__init__()
#         self.num_outputs = num_outputs
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight("weights",
#                                       shape=[int(input_shape[-1]),
#                                              self.num_outputs])
#         print(type(self.kernel))
#
#     def call(self, inputs):
#         return tf.sparce.sparse_tensor_dense_matmul(inputs, self.kernel)
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


def sparce_to_tensor(array):
    indicies, values, dense_shape = sparse_to_tuple(array)


def normalize_adj_np(adj):
    adj_ = adj + np.identity(adj.shape[0])
    degrees = np.count_nonzero(adj_, axis=0) ** -0.5
    D = np.diag(degrees)
    adj_normalized = np.matmul(np.matmul(adj_, D).T, D)
    return adj_normalized


# taken from ...
def normalize_adj_sp(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized


def random_adj(shape):
    adj_init = np.random.randint(2, size=(shape, shape))
    for x in range(4):
        adj_init[x, x] = 0
    adj_tri = np.tril(adj_init)
    adj = adj_tri + adj_tri.T
    return adj


def actual_normalize(adj):
    adj_ = adj + np.identity(adj.shape[0])
    degrees = np.count_nonzero(adj_, axis=0) ** -0.5
    D = np.diag(degrees)
    adj_normalized = np.matmul(np.matmul(D, adj_), D)
    return adj_normalized


def actual_normalize_sp(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    counts = []
    for i in range(adj.shape[0]):
        _, cols = adj_[i, :].nonzero()
        counts.append(len(cols))
    D = sp.diags(np.array(counts).astype('float32') ** -0.5)
    adj_normalized = D.dot(adj_).dot(D)
    return adj_normalized


def dense_lae(adj_norm, adj, num_epochs, learning_rate):
    r = (adj != 0).astype(int).astype(adj.dtype)
    weight = float(adj.shape[0] * adj.shape[0] - r.sum()) / r.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * r.shape[0] - r.sum()) * 2)

    A = tf.Variable(adj_norm.astype('float32'))
    B = tf.Variable(adj.astype('float32'))
    W = weight_variable_glorot(adj_norm.shape[0], 10)

    for x in range(num_epochs):
        with tf.GradientTape() as tape:
            tape.watch(W)
            Z = tf.linalg.matmul(A, W)
            An = tf.math.sigmoid(tf.linalg.matmul(Z, tf.transpose(Z)))
            loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(B, An, weight))
            if x % 5 == 0:
                print(x, loss)
        grads = tape.gradient(loss, W)
        W = W - learning_rate * grads


def sparce_lae(adj_norm, adj, num_epochs, learning_rate):
    components = 10
    r = (adj != 0).astype(int).astype(adj.dtype)
    weight = float(adj.shape[0] * adj.shape[0] - r.sum()) / r.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * r.shape[0] - r.sum()) * 2)

    A = tf.SparseTensor(*sparse_to_tuple(adj_norm.astype('float32')))
    B = tf.sparse.to_dense(tf.SparseTensor(*sparse_to_tuple(adj.astype('float32'))))
    W = weight_variable_glorot(adj_norm.shape[0], components)

    for x in range(num_epochs):
        with tf.GradientTape() as tape:
            tape.watch(W)
            Z = tf.sparse.sparse_dense_matmul(A, W)
            An = tf.math.sigmoid(tf.linalg.matmul(Z, tf.transpose(Z)))

            loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(B, An, weight))
            if x % 25 == 0:
                print(x, loss)
        grads = tape.gradient(loss, W)
        W = W - learning_rate * grads


def tf1(adj_norm, adj, num_epochs, learning_rate):
    components = 10
    r = (adj != 0).astype(int).astype(adj.dtype)
    weight = float(adj.shape[0] * adj.shape[0] - r.sum()) / r.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * r.shape[0] - r.sum()) * 2)

    adj_norm_tensor = tf.SparseTensor(*sparse_to_tuple(adj_norm.astype('float32')))
    adj_tensor = tf.sparse.to_dense(tf.SparseTensor(*sparse_to_tuple(adj.astype('float32'))))
    with tf.compat.v1.variable_scope('vars'):
        weight_tensor = weight_variable_glorot(adj_norm.shape[0], components,name='weights')


    # def loss_func
    Z = tf.sparse.sparse_dense_matmul(adj_norm_tensor, weight_tensor)
    An = tf.math.sigmoid(tf.linalg.matmul(Z, tf.transpose(Z)))

    loss = lambda: norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(adj_tensor, An, weight))
    #
    #
    #
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    opt_op = opt.minimize(loss, var_list=[weight_tensor])
    train_summary_writer = tf.summary.create_file_writer('log_dir')
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(num_epochs):
        outs = sess.run([opt_op, loss])
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', 5, step=epoch)
        train_summary_writer.flush()

        # if epoch % 25 == 0:
        #     print(f'Epoch: {str(epoch).rjust(len(str(num_epochs-1)))}, Loss: {outs[1]} ')

    #     opt.minimize(loss,[weight_tensor])
    #     print(loss())


def main():
    tf.random.set_seed(1)

    learning_rate = .5
    adj = np.load('adj500.npy')
    print(adj.shape[0])
    adj = adj / np.argmax(adj)

    # print(actual_normalize(adj))
    dense = False
    # if dense:
    # s = perf_counter()
    # adj_norm = actual_normalize(adj)
    # dense_lae(adj_norm, adj, 25, learning_rate)
    # print(perf_counter()-s)
    # else:
    # s = perf_counter()
    adjs = sp.csr_matrix(adj)
    adj_norm_sp = actual_normalize_sp(adjs)
    # sparce_lae(adj_norm_sp, adjs, 500, learning_rate)
    # print(perf_counter() - s)
    tf1(adj_norm_sp, adjs, 500, learning_rate)


if __name__ == '__main__':
    main()
