import numpy as np
import pandas as pd
import arena_util as au
import time
from collections import Counter
import numba as nb
from numba import njit
from numba import jitclass
from numba import int32, float32, types, boolean

'''
genre = au.load_json("./res/genre_gn_all.json")
song = au.load_json("./res/song_meta.json")

train_virtual = au.load_json("./arena_data/orig/train.json")
val_virtual = au.load_json("./arena_data/questions/val.json")
train = au.load_json("./res/train.json")
val = au.load_json("./res/val.json")
'''

'''
spec_MatrixCoo = [
        ('row', int32), 
        ('col', int32), 
        ('prev_N', int32), 
        ('N', int32), 
        ('data_user', int32[:]),
        ('data_item', int32[:]),
        ('data_score', float32[:]),
]
'''

class MatrixCoo:
    def __init__(self):
        self.row = self.col = self.N = self.prev_N = 0
        self.data_user = np.array([], dtype=np.int32)
        self.data_item = np.array([], dtype=np.int32)
        self.data_score = np.array([], dtype=np.float32)

    def read(self, fname):
        f = open(fname, 'r')
        r, c, n = f.readline().split()
        self.row += int(r)
        self.col = int(c)
        self.prev_N = self.N
        self.N += int(n)
        self.data_user.resize(self.N)
        self.data_item.resize(self.N)
        self.data_score.resize(self.N)
        for i in range(self.prev_N, self.N):
            self.data_user[i], self.data_item[i], self.data_score[i] = f.readline().split()
        f.close()

'''
spec_eALS = [
        ('num_factor', int32), 
        ('max_epoch', int32), 
        ('ld', float32), 
        ('confidence', float32), 
        ('diff_threshold', float32), 
        ('verbose', boolean), 
        ('assigned', boolean), 
        ('fname_train', nb.types.string),
        ('fname_val', nb.types.string),
        ('coodata', MatrixCoo.class_type.instance_type),
        ('val_user', int32),
        ('num_user', int32),
        ('num_item', int32),
        ('X', float32[:, :]),
        ('X_current', float32[:, :]),
        ('Y', float32[:, :]),
        ('Y_current', float32[:, :]),
]
'''

class eALS:
    def __init__(self, _num_factor, _max_epoch, _ld, _confidence, _diff_threshold, _verbose, _fname_train, _fname_val):
        self.num_factor = _num_factor
        self.max_epoch = _max_epoch
        self.ld = _ld
        self.confidence = _confidence
        self.diff_threshold = _diff_threshold
        self.verbose = _verbose
        self.fname_train = _fname_train
        self.fname_val = _fname_val
        self.coodata = MatrixCoo()
        self.assigned = False

    def loadData(self):
        self.coodata.read(self.fname_train)
        num_tmp = self.coodata.row
        self.coodata.read(self.fname_val)
        self.val_user = self.coodata.row - num_tmp
        self.num_user = self.coodata.row
        self.num_item = self.coodata.col
        print("Matrix density : {}".format(self.coodata.N / self.num_user / self.num_item, ".8f"))
        print("N : {}".format(self.coodata.N))
        print("# of users : {}, # of items : {}".format(self.num_user, self.num_item))

    def train(self):
        self.X = np.random.rand(self.num_user, self.num_factor)
        self.X_current = np.random.rand(self.num_user, self.num_factor)
        self.Y = np.random.rand(self.num_item, self.num_factor)
        self.Y_current = np.random.rand(self.num_item, self.num_factor)
        self.getRhat()
        for i in range(self.max_epoch):
            print("Epoch {} start".format(i+1))
            self.train_epoch()
            diff = self.checkConvergence()
            if(self.verbose):
                if(diff < self.diff_threshold):
                    print("\nEpoch : {}".format(i+1))
                    print("X and Y converged.")
                    break
                if(i == self.max_epoch - 1):
                    print("\nEpoch : {}".format(i+1))
                    print("Reached maximum iteration")

    def getRhat(self):
        self.rui_hat = [None] * self.num_user
        self.col_num = [0] * self.num_user
        self.riu_hat = [None] * self.num_item
        self.row_num = [0] * self.num_item
        for i in range(self.coodata.N):
            r, c = self.coodata.data_user[i], self.coodata.data_item[i]
            if(self.rui_hat[r] == None):
                self.rui_hat[r] = [[self.X[r].dot(self.Y[c]), c, None]]
            else:
                self.rui_hat[r].append([self.X[r].dot(self.Y[c]), c, None])
            if(self.riu_hat[c] == None):
                self.riu_hat[c] = [[self.rui_hat[r][self.col_num[r]][0], r, None]]
            else:
                self.riu_hat[c].append([self.rui_hat[r][self.col_num[r]][0], r, None])
            self.rui_hat[r][self.col_num[r]][2] = (c, self.row_num[c])
            self.riu_hat[c][self.row_num[c]][2] = (r, self.col_num[r])
            self.row_num[c] += 1
            self.col_num[r] += 1

    def checkConvergence(self):
        if self.assigned:
            diff_X = np.linalg.norm(self.X - self.X_current)
            diff_Y = np.linalg.norm(self.Y - self.Y_current)
            print("Difference(Frobenius norm square) - \
                  X : {}, Y : {}".format(diff_X**2, diff_Y**2, ".6f"))
        self.X_current = self.X
        self.Y_current = self.Y
        self.assigned = True
        return max(diff_X**2, diff_Y**2)

    def train_epoch(self):
        #train user factors
        starttime = time.time()
        Sq = self.Y.T.dot(self.Y)
        for u in range(self.num_user):
            for f in range(self.num_factor):
                rui_hat_f = np.zeros(self.col_num[u], dtype=np.float32)
                numerator = self.X[u][f] * Sq[f][f] - self.X[u].dot(Sq[f])
                denominator = Sq[f][f] + self.ld
                for i_ind in range(self.col_num[u]):
                    qif = self.Y[self.rui_hat[u][i_ind][1]][f]
                    rui_hat_f[i_ind] = self.rui_hat[u][i_ind][0] - self.X[u][f] * qif
                    denominator += self.confidence * qif * qif
                self.X[u][f] = numerator / denominator
                for i_ind in range(self.col_num[u]):
                    qif = self.Y[self.rui_hat[u][i_ind][1]][f]
                    self.rui_hat[u][i_ind][0] = self.X[u][f] * qif + rui_hat_f[i_ind]
        for u in range(self.num_user):
            for i_ind in range(self.col_num[u]):
                i, j = self.rui_hat[u][i_ind][2]
                self.riu_hat[i][j] = self.rui_hat[u][i_ind][0]
        print("Updating X took {}s".format(time.time() - starttime, ".2f"))
        # train item factors
        starttime = time.time()
        Sp = self.X.T.dot(self.X)
        for i in range(self.num_item):
            for f in range(self.num_factor):
                riu_hat_f = np.zeros(self.row_num[i], dtype=np.float32)
                numerator = self.Y[i][f] * Sp[f][f] - self.Y[i].dot(Sp[f])
                denominator = Sp[f][f] + self.ld
                for u_ind in range(self.row_num[i]):
                    puf = self.X[self.riu_hat[i][u_ind][1]][f]
                    riu_hat_f[u_ind] = self.riu_hat[i][u_ind][0] - self.Y[i][f] * puf
                    denominator += self.confidence * puf * puf
                self.Y[i][f] = numerator / denominator
                for u_ind in range(self.row_num[i]):
                    puf = self.X[self.riu_hat[i][u_ind][1]][f]
                    self.riu_hat[i][u_ind][0] = self.Y[i][f] * puf + riu_hat_f[u_ind]
        for i in range(self.num_item):
            for u_ind in range(row_num[i]):
                i, j = self.riu_hat[i][u_ind][2]
                self.rui_hat[i][j] = self.riu_hat[i][u_ind][0]
        print("Updating Y took {}s".format(time.time() - starttime, ".2f"))


cfg = au.load_json('./cfg.json')
eals = eALS(cfg['num_factor'], cfg['max_epoch'], cfg['lambda'], cfg['confidence'], \
            cfg['diff_threshold'], cfg['verbose'], cfg['fname_train'], cfg['fname_val'])
eals.loadData()
eals.train()

f = open('./userpy.txt', 'w')
for i in range(eals.num_user - eals.val_user, eals.num_user):
    logit = eals.X[i]
    f.write(logit)
f.close()
f = open('./songpy.txt', 'w')
for i in range(eals.num_item):
    logit = eals.Y[i]
    f.write(logit)
f.close()
