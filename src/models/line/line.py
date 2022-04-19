import os
import pathlib
import subprocess
import sys
import time
from os.path import join

import numpy as np
import scipy.sparse as sp

script_dir = pathlib.Path(__file__).parent.resolve()
print(script_dir)


def adj_to_txt(adj, filename='out.txt'):
    with open(filename, 'w') as outfile:
        tmp_adj = sp.coo_matrix(adj)
        for row, col, data in zip(tmp_adj.row, tmp_adj.col, tmp_adj.data):
            outfile.write(f'{row}\t{col}\t{data:0.5f}\n')


def print_call(args, fail_limit=4):
    print(' '.join(args))
    done = False
    fails = 0
    while not done:
        try:
            subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stdout)
            done = True
        except subprocess.CalledProcessError:
            fails += 1
            if fails < fail_limit:
                print(f'Fail(s):{fails}\nRetrying')
                time.sleep(2)
            else:
                print(f'\nFailed {fails} time(s) which exceeds the limit.\nExiting...')
                exit(1)


def get_line_comps(adj, stem='tmp', depth=2, threshold=1000, size=64, negative=5, samples=2500, threads=20):
    adj_to_txt(adj, join(script_dir, stem + '_adj.txt'))
    print_call([join(script_dir, 'reconstruct'), '-train', join(script_dir, stem + '_adj.txt'), '-output',
                join(script_dir, stem + '_dense.txt'), '-depth', str(depth), '-threshold', str(threshold)])

    print_call(
        [join(script_dir, 'line'), '-train', join(script_dir, stem + '_dense.txt'), '-output',
         join(script_dir, 'vec_1st_wo_norm.txt'), '-binary', '1', '-size', str(size), '-order', '1', '-negative',
         str(negative), '-samples', str(samples), '-threads', str(threads)])

    print_call(
        [join(script_dir, 'line'), '-train', join(script_dir, stem + '_dense.txt'), '-output',
         join(script_dir, 'vec_2nd_wo_norm.txt'), '-binary', '1', '-size', str(size), '-order', '2', '-negative',
         str(negative), '-samples', str(samples), '-threads', str(threads)])

    print_call([join(script_dir, 'normalize'), '-input', join(script_dir, 'vec_1st_wo_norm.txt'), '-output',
                join(script_dir, 'vec_1st.txt'), '-binary', '1'])

    print_call([join(script_dir, 'normalize'), '-input', join(script_dir, 'vec_2nd_wo_norm.txt'), '-output',
                join(script_dir, 'vec_2nd.txt'), '-binary', '1'])

    print_call([join(script_dir, 'concatenate'), '-input1', join(script_dir, 'vec_1st.txt'), '-input2',
                join(script_dir, 'vec_2nd.txt'), '-output', join(script_dir, 'vec_all.txt'), '-binary', '0'])

    data = []
    with open(join(script_dir, 'vec_all.txt'), 'r') as infile:
        infile.readline()

        while True:
            l = infile.readline()
            if not l:
                break
            i = l.index(' ')
            id = int(l[:i])
            arr = np.fromstring(l[i + 1:], np.float32, sep=' ')
            data.append((id, arr))

    data.sort(key=lambda x: x[0])
    comps = np.row_stack([x[1] for x in data])
    # for file in [stem + '_adj.txt', stem + '_dense.txt', 'vec_1st_wo_norm.txt', 'vec_2nd_wo_norm.txt', 'vec_1st.txt',
    #              'vec_2nd.txt', 'vec_all.txt']:
    #     os.remove(join(script_dir, file))
    print(np.isnan(comps).any())
    return comps


if __name__ == '__main__':
    adj = sp.load_npz('/home/belfner/Documents/Spatial_Temporal_Mining/Project/src/tests/adj_1507.npz')
    comps = get_line_comps(adj)
    print(comps.shape)
    print(comps[0].shape)
