import numpy as np
import pandas as pd
import arena_util as au
import time
from collections import Counter
import numba as nb
from numba import njit

def read_mm(fname):
    f = open(fname, 'r')
    r, c, n = f.readline().split()
    global row, prev_row, col, N, prev_N
    prev_row = row
    row += int(r)
    col = int(c)
    prev_N = int(N)
    N += int(n)
    data_user.resize(N)
    data_item.resize(N)
    data_score.resize(N)
    for i in range(prev_N, N):
        data_user[i], data_item[i], data_score[i] = f.readline().split()
        data_user[i] += prev_row
    f.close()

#@njit
def getRhat():
    for i in range(N):
        r, c = data_user[i], data_item[i]
        rui_hat[r].append([P[r].dot(Q[c]), c, -1])
        riu_hat[c].append([rui_hat[r][col_num[r]][0], r, -1])
        rui_hat[r][-1][2] = row_num[c]
        riu_hat[c][-1][2] = col_num[r]
        row_num[c] += 1
        col_num[r] += 1

#@njit
def checkConvergence():
    global assigned, P_current, Q_current
    diff_P = diff_Q = -1
    if assigned:
        diff_P = np.linalg.norm(P - P_current)
        diff_Q = np.linalg.norm(Q - Q_current)
        print("Difference(Frobenius norm) - \
              P : %.10f, Q : %.10f" % (diff_P, diff_Q))
    P_current = P.copy()
    Q_current = Q.copy()
    assigned = True
    return max(diff_P, diff_Q)

#@njit
def train_epoch():
    #train user factors
    starttime = time.time()
    Sq = Q.T.dot(Q)
    for u in range(num_user):
        for f in range(num_factor):
            rui_hat_f = np.zeros(col_num[u], dtype=np.float32)
            numerator = P[u][f] * Sq[f][f] - P[u].dot(Sq[f])
            denominator = Sq[f][f] + ld
            for i_ind in range(col_num[u]):
                qif = Q[rui_hat[u][i_ind][1]][f]
                rui_hat_f[i_ind] = rui_hat[u][i_ind][0] - P[u][f] * qif
                numerator += (1 + confidence - confidence * rui_hat_f[i_ind]) * qif
                denominator += confidence * qif * qif
            P[u][f] = numerator / denominator
            for i_ind in range(col_num[u]):
                qif = Q[rui_hat[u][i_ind][1]][f]
                rui_hat[u][i_ind][0] = P[u][f] * qif + rui_hat_f[i_ind]
                c, u_ind = rui_hat[u][i_ind][1:]
                riu_hat[c][u_ind][0] = rui_hat[u][i_ind][0]
    print("Updating P took %.2fs" % (time.time() - starttime))

    # train item factors
    starttime = time.time()
    Sp = P.T.dot(P)
    for i in range(num_item):
        for f in range(num_factor):
            riu_hat_f = np.zeros(row_num[i], dtype=np.float32)
            numerator = Q[i][f] * Sp[f][f] - Q[i].dot(Sp[f])
            denominator = Sp[f][f] + ld
            for u_ind in range(row_num[i]):
                puf = P[riu_hat[i][u_ind][1]][f]
                riu_hat_f[u_ind] = riu_hat[i][u_ind][0] - Q[i][f] * puf
                numerator += (1 + confidence - confidence * riu_hat_f[u_ind]) * puf
                denominator += confidence * puf * puf
            Q[i][f] = numerator / denominator
            for u_ind in range(row_num[i]):
                puf = P[riu_hat[i][u_ind][1]][f]
                riu_hat[i][u_ind][0] = Q[i][f] * puf + riu_hat_f[u_ind]
                r, i_ind = riu_hat[i][u_ind][1:]
                rui_hat[r][i_ind][0] = riu_hat[i][u_ind][0]
    print("Updating Q took %.2fs" % (time.time() - starttime))

if __name__ == "__main__":

    # hyperparameter setting
    cfg = au.load_json('./cfg.json')

    # N : number of nonzero elements
    row = col = N = prev_N = 0
    data_user = np.array([], dtype=np.int32)
    data_item = np.array([], dtype=np.int32)
    data_score = np.array([], dtype=np.float32)

    num_factor = cfg['num_factor']
    max_epoch = cfg['max_epoch']
    ld = cfg['lambda']
    confidence = cfg['confidence']
    diff_threshold = cfg['diff_threshold']
    verbose = cfg['verbose']
    fname_train = cfg['fname_train']
    fname_val = cfg['fname_val']
    assigned = False

    read_mm(fname_train)
    num_user_train = row
    read_mm(fname_val)
    num_user_val = row - num_user_train
    num_user = row
    num_item = col

    print("Matrix density : {}".format(N / num_user / num_item))
    print("N : {}".format(N))
    print("# of users : {}, # of items : {}".format(num_user, num_item))

    # initialize X and Y (-1, 1)
    P = np.random.rand(num_user, num_factor)
    P = P * 2 - 1
    P_current = P
    Q = np.random.rand(num_item, num_factor)
    Q = Q * 2 - 1
    Q_current = Q

    rui_hat = [[] for _ in range(num_user)]
    col_num = [0] * num_user
    riu_hat = [[] for _ in range(num_item)]
    row_num = [0] * num_item
    getRhat()

    for i in range(max_epoch):
        print("Epoch {} start".format(i+1))
        train_epoch()
        diff = checkConvergence()
        if(verbose):
            if(diff > 0 and diff < diff_threshold):
                print("\nEpoch : {}".format(i+1))
                print("P and Q converged.")
                break
            if(i == max_epoch - 1):
                print("\nEpoch : {}".format(i+1))
                print("Reached maximum iteration")

    with open('./user.txt', 'w') as f:
        for i in range(num_user - num_user_val, num_user):
            np.savetxt(f, P[i], newline = " ", fmt="%.8lf")
            f.write("\n")
    f.close()
    with open('./item.txt', 'w') as f:
        for i in range(num_item):
            np.savetxt(f, Q[i], newline = " ", fmt="%.8lf")
            f.write("\n")
    f.close()
