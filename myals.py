""" 
Revised Implicit Alternating Least Squares 
Reference - https://github.com/benfred/implicit
Only GPU version available 
"""
import functools
import heapq
import logging
import time
import torch
import pickle

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from tqdm.auto import tqdm

import implicit.cuda

from implicit.als import _als
from implicit.evaluation import train_test_split
from implicit.recommender_base import MatrixFactorizationBase
from implicit.utils import check_blas_config, nonzeros

log = logging.getLogger("implicit")

class MyAlternatingLeastSquares(MatrixFactorizationBase):

    """ My Alternating Least Squares

    A revised version of Alternating Least Squares of Implicit library.
    Diffrence is that myals does not find similar playlists but predicted items 
    directly by matrix multiplication.

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_native : bool, optional
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    use_gpu : bool, optional
        Fit on the GPU if available, default is to run on GPU only if available
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    validate_step : int, optional
        if validate_step > 0, periodically (per validate_step) validates
        the model. validation dataset must be given to argument of
        `fit`method.
        if validate_step <= 0, no validation is done.
    validate_N : int, optional
        size of truncation of validation metric.
        it has no meaning when validate_step <= 0.
    validate_proportion : float, optional
        the portion of validation matrix to total matrix. default value
        to be 0.05
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(self, num_song=707989, num_tag=30197, factors=100, 
                 regularization=0.01, dtype=np.float32, use_native=True, 
                 use_cg=True, use_gpu=implicit.cuda.HAS_CUDA,
                 iterations=15, calculate_training_loss=False,
                 validate_step=-1, validate_N=30, validate_proportion=0.05,
                 num_threads=0, test_fname="./res/test.json"):
        super(MyAlternatingLeastSquares, self).__init__()
        # currently there are some issues when training on the GPU when some of the warps
        # don't have full factors. Round up to be warp aligned.
        # TODO: figure out where the issue is (best guess is in the
        # the 'dot' function in 'implicit/cuda/utils/cuh)
        if use_gpu and factors % 32:
            padding = 32 - factors % 32
            log.warning("GPU training requires factor size to be a multiple of 32."
                        " Increasing factors from %i to %i.", factors, factors + padding)
            factors += padding

        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization
        self.num_song = num_song
        self.num_tag = num_tag
        self.test_fname = test_fname

        # options on how to fit the model
        self.dtype = dtype
        self.use_native = use_native
        self.use_cg = use_cg
        self.use_gpu = use_gpu
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.num_threads = num_threads
        self.fit_callback = None
        self.cg_steps = 3
        if validate_step < 0:
            self.use_validation = False
        else:
            self.use_validation = True
        self.validate_step = validate_step
        self.validate_N = validate_N
        self.validate_proportion = validate_proportion

        # cache for item factors squared
        self._YtY = None

        check_blas_config()

    def fit(self, item_users, show_progress=True):
        """ Factorizes the item_users matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The item_users matrix does double duty here. It defines which items are liked by which
        users (P_iu in the original paper), as well as how much confidence we have that the user
        liked the item (C_iu).

        The negative items are implicitly defined: This code assumes that non-zero items in the
        item_users matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.

        Parameters
        ----------
        item_users: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the item, the columns are the users that liked that item,
            and the value is the confidence that the user liked the item.
        vali_item_users: csr_matrix
            Same format with item_users. It is used to validate the model.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        """

        if self.use_validation is True:
            item_users, vali_item_users = train_test_split(
                item_users, 1.0 - self.validate_proportion)
            vali_user_items = vali_item_users.T
        else:
            vali_user_items = None

        Ciu = item_users
        if not isinstance(Ciu, scipy.sparse.csr_matrix):
            s = time.time()
            log.debug("Converting input to CSR format")
            Ciu = Ciu.tocsr()
            log.debug("Converted input to CSR in %.3fs", time.time() - s)

        if Ciu.dtype != np.float32:
            Ciu = Ciu.astype(np.float32)

        s = time.time()
        Cui = Ciu.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = np.random.rand(users, self.factors).astype(self.dtype) * 0.01
        if self.item_factors is None:
            self.item_factors = np.random.rand(items, self.factors).astype(self.dtype) * 0.01

        log.debug("Initialized factors in %s", time.time() - s)

        # invalidate cached norms and squared factors
        self._item_norms = None
        self._YtY = None

        if self.use_gpu:
            return self._fit_gpu(Ciu, Cui, vali_user_items, show_progress)

        solver = self.solver

        log.debug("Running %i ALS iterations", self.iterations)
        
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            # alternate between learning the user_factors from the item_factors and vice-versa
            for iteration in range(self.iterations):
                s = time.time()
                solver(Cui, self.user_factors, self.item_factors, self.regularization,
                       num_threads=self.num_threads)
                solver(Ciu, self.item_factors, self.user_factors, self.regularization,
                       num_threads=self.num_threads)
                progress.update(1)

                if self.calculate_training_loss:
                    loss = _als.calculate_loss(Cui, self.user_factors, self.item_factors,
                                               self.regularization, num_threads=self.num_threads)
                    progress.set_postfix({"loss": loss})

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

                if self.use_validation and ((iteration + 1) % self.validate_step) == 0:
                    vali_res = self.validate(Cui, vali_user_items, self.validate_N)
                    log.info(
                        "[iter %d] Precision %0.4f MAP %0.4f NDCG %0.4f AUC %0.4f" %
                        (iteration,
                         vali_res["precision"],
                         vali_res["map"],
                         vali_res["ndcg"],
                         vali_res["auc"]))
                
        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)



    def _fit_gpu(self, Ciu_host, Cui_host, vali_user_items, show_progress=True):
        """ specialized training on the gpu. copies inputs to/from cuda device """
        if not implicit.cuda.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        if self.dtype == np.float64:
            log.warning("Factors of dtype float64 aren't supported with gpu fitting. "
                        "Converting factors to float32")
            self.item_factors = self.item_factors.astype(np.float32)
            self.user_factors = self.user_factors.astype(np.float32)

        Ciu = implicit.cuda.CuCSRMatrix(Ciu_host)
        Cui = implicit.cuda.CuCSRMatrix(Cui_host)
        X = implicit.cuda.CuDenseMatrix(self.user_factors.astype(np.float32))
        Y = implicit.cuda.CuDenseMatrix(self.item_factors.astype(np.float32))

        solver = implicit.cuda.CuLeastSquaresSolver(self.factors)
        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                s = time.time()
                solver.least_squares(Cui, X, Y, self.regularization, self.cg_steps)
                solver.least_squares(Ciu, Y, X, self.regularization, self.cg_steps)
                progress.update(1)

                if self.calculate_training_loss:
                    loss = solver.calculate_loss(Cui, X, Y, self.regularization)
                    progress.set_postfix({"loss": loss})

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

                if self.use_validation and ((iteration + 1) % self.validate_step) == 0:
                    vali_res = self.validate(Cui, vali_user_items, self.validate_N)
                    log.info(
                        "[iter %d] Precision %0.4f MAP %0.4f NDCG %0.4f AUC %0.4f" %
                        (iteration,
                         vali_res["precision"],
                         vali_res["map"],
                         vali_res["ndcg"],
                         vali_res["auc"]))
        
        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

        mask = []
        test = pd.read_json(self.test_fname, encoding='UTF-8')
        for i, q in test.iterrows():
            if(q['songs'] != [] or q['tags'] != []):
                mask.append(q['id'])
        del test

        X.to_host(self.user_factors)
        Y.to_host(self.item_factors)
        
        X_tensor = torch.FloatTensor(self.item_factors[mask,:])
        Y1_tensor = torch.FloatTensor(self.user_factors.T[:,:self.num_song])
        Y2_tensor = torch.FloatTensor(self.user_factors.T[:,self.num_song:self.num_song+self.num_tag])
        del self.item_factors
        del self.user_factors
        f = open("./predicted_songs.pkl", 'wb')
        g = open("./predicted_tags.pkl", 'wb')
        songs = []
        tags = []
        x = X_tensor.to("cuda")
        del X_tensor
        y1 = Y1_tensor.to("cuda")
        del Y1_tensor
        y2 = Y2_tensor.to("cuda")
        del Y2_tensor        
        for i in tqdm(range(x.shape[0])):
            results = torch.mm(x[i:(i+1),], y1)
            results2 = torch.mm(x[i:(i+1),], y2)
            _, indices_song = torch.sort(results, descending=True)
            _, indices_tag = torch.sort(results2, descending=True)
            songs.append(indices_song[0,:200].cpu().numpy())
            tags.append(indices_tag[0,:20].cpu().numpy())
                                
        pickle.dump(songs, f)
        pickle.dump(tags, g)
        f.close()
        g.close()

    def recalculate_user(self, userid, user_items):
        return user_factor(self.item_factors, self.YtY,
                           user_items.tocsr(), userid,
                           self.regularization, self.factors)

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):
        """ Provides explanations for why the item is liked by the user.

        Parameters
        ---------
        userid : int
            The userid to explain recommendations for
        user_items : csr_matrix
            Sparse matrix containing the liked items for the user
        itemid : int
            The itemid to explain recommendations for
        user_weights : ndarray, optional
            Precomputed Cholesky decomposition of the weighted user liked items.
            Useful for speeding up repeated calls to this function, this value
            is returned
        N : int, optional
            The number of liked items to show the contribution for

        Returns
        -------
        total_score : float
            The total predicted score for this user/item pair
        top_contributions : list
            A list of the top N (itemid, score) contributions for this user/item pair
        user_weights : ndarray
            A factorized representation of the user. Passing this in to
            future 'explain' calls will lead to noticeable speedups
        """
        # user_weights = Cholesky decomposition of Wu^-1
        # from section 5 of the paper CF for Implicit Feedback Datasets
        user_items = user_items.tocsr()
        if user_weights is None:
            A, _ = user_linear_equation(self.item_factors, self.YtY,
                                        user_items, userid,
                                        self.regularization, self.factors)
            user_weights = scipy.linalg.cho_factor(A)
        seed_item = self.item_factors[itemid]

        # weighted_item = y_i^t W_u
        weighted_item = scipy.linalg.cho_solve(user_weights, seed_item)

        total_score = 0.0
        h = []
        for i, (itemid, confidence) in enumerate(nonzeros(user_items, userid)):
            if confidence < 0:
                continue

            factor = self.item_factors[itemid]
            # s_u^ij = (y_i^t W^u) y_j
            score = weighted_item.dot(factor) * confidence
            total_score += score
            contribution = (score, itemid)
            if i < N:
                heapq.heappush(h, contribution)
            else:
                heapq.heappushpop(h, contribution)

        items = (heapq.heappop(h) for i in range(len(h)))
        top_contributions = list((i, s) for s, i in items)[::-1]
        return total_score, top_contributions, user_weights

    @property
    def solver(self):
        if self.use_cg:
            solver = _als.least_squares_cg if self.use_native else least_squares_cg
            return functools.partial(solver, cg_steps=self.cg_steps)
        return _als.least_squares if self.use_native else least_squares

    @property
    def YtY(self):
        if self._YtY is None:
            Y = self.item_factors
            self._YtY = Y.T.dot(Y)
        return self._YtY


def my_alternating_least_squares(Ciu, factors, **kwargs):
    """ factorizes the matrix Cui using an implicit alternating least squares
    algorithm. Note: this method is deprecated, consider moving to the
    AlternatingLeastSquares class instead

    """
    log.warning("This method is deprecated. Please use the AlternatingLeastSquares"
                " class instead")

    model = MyAlternatingLeastSquares(factors=factors, **kwargs)
    model.fit(Ciu)
    return model.item_factors, model.user_factors


def least_squares(Cui, X, Y, regularization, num_threads=0):
    """ For each user in Cui, calculate factors Xu for them
    using least squares on Y.

    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        X[u] = user_factor(Y, YtY, Cui, u, regularization, n_factors)


def user_linear_equation(Y, YtY, Cui, u, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)

    # accumulate YtCuY + regularization*I in A
    A = YtY + regularization * np.eye(n_factors)

    # accumulate YtCuPu in b
    b = np.zeros(n_factors)

    for i, confidence in nonzeros(Cui, u):
        factor = Y[i]

        if confidence > 0:
            b += confidence * factor
        else:
            confidence *= -1

        A += (confidence - 1) * np.outer(factor, factor)
    return A, b


def user_factor(Y, YtY, Cui, u, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    A, b = user_linear_equation(Y, YtY, Cui, u, regularization, n_factors)
    return np.linalg.solve(A, b)


def least_squares_cg(Cui, X, Y, regularization, num_threads=0, cg_steps=3):
    users, factors = X.shape
    YtY = Y.T.dot(Y) + regularization * np.eye(factors, dtype=Y.dtype)

    for u in range(users):
        # start from previous iteration
        x = X[u]

        # calculate residual error r = (YtCuPu - (YtCuY.dot(Xu)
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            if confidence > 0:
                r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]
            else:
                confidence *= -1
                r += - (confidence - 1) * Y[i].dot(x) * Y[i]

        p = r.copy()
        rsold = r.dot(r)
        if rsold < 1e-20:
            continue

        for it in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                if confidence < 0:
                    confidence *= -1

                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if rsnew < 1e-20:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

