"""
An example based off the MovieLens 20M dataset.
This code will automatically download a HDF5 version of this
dataset when first run. The original dataset can be found here:
https://grouplens.org/datasets/movielens/.
Since this dataset contains explicit 5-star ratings, the ratings are
filtered down to positive reviews (4+ stars) to construct an implicit
dataset
"""

# movie = playlist

from __future__ import print_function

import argparse
import codecs
import logging
import time
import arena_data as au
from scipy.sparse import csr_matrix

import numpy as np
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

log = logging.getLogger("implicit")


def calculate_similar_movies(output_filename,
                             model_name="als", min_rating=0.0,
                             variant='20m', K = 20, factors = 50):

    start = time.time()
    
    row = []
    col = []
    rating = []
    mask = []
    with open("./coo.txt", 'r') as f :
        r,c,N = f.readline().split()
        mask = [False] * int(r)
        titles = np.arange(int(r))
        for _ in range(int(N)):
            a,b,d,m = f.readline().split()
            if m == '1':
                mask[int(a)] = True
            row.append(int(a))
            col.append(int(b))
            rating.append(float(d))
    f.close()
    titles = np.array([str(i) for i in titles])
    ratings = csr_matrix((np.array(rating, dtype=np.float32), (np.array(row), np.array(col))), shape = (int(r), int(c)))

    
    ratings.data = np.ones(len(ratings.data))
    log.info("read data file in %s", time.time() - start)

    # generate a recommender model based off the input params
    if model_name == "als":
        model = AlternatingLeastSquares(factors = factors)

        # lets weight these models by bm25weight.
        log.debug("weighting matrix by bm25_weight")
        ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()

    elif model_name == "cosine":
        print("Cosine Model by K = {0}".format(K))
        model = CosineRecommender(K = K)

    elif model_name == "bm25":
        model = BM25Recommender(K = K, B=0.2)

    else:
        raise NotImplementedError("TODO: model %s" % model_name)

    # train the model
    log.debug("training model %s", model_name)
    start = time.time()
    model.fit(ratings)
    log.debug("trained model '%s' in %s", model_name, time.time() - start)
    log.debug("calculating top movies")

    user_count = np.ediff1d(ratings.indptr)
    to_generate = sorted(np.arange(len(titles)), key=lambda x: -user_count[x])
    
    log.debug("calculating similar playlists")
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for playid in to_generate:
                # if this movie has no ratings, skip over (for instance 'Graffiti Bridge' has
                # no ratings > 4 meaning we've filtered out all data for it.
                if ratings.indptr[playid] != ratings.indptr[playid + 1] and mask[playid]:
                    title = titles[playid]
                    for other, score in model.similar_items(playid, 201):
                        o.write("%s %s\n" % (title, titles[other]))
                progress.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates related movies from the MovieLens 20M "
                                     "dataset (https://grouplens.org/datasets/movielens/20m/)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output', type=str, default='similar-playlist.txt',
                        dest='outputfile', help='output file name')
    parser.add_argument('--model', type=str, default='cosine',
                        dest='model', help='model to calculate (als/bm25/tfidf/cosine)')
    parser.add_argument('--variant', type=str, default='20m', dest='variant',
                        help='Whether to use the 20m, 10m, 1m or 100k movielens dataset')
    parser.add_argument('--min_rating', type=float, default=0.0, dest='min_rating',
                        help='Minimum rating to assume that a rating is positive')
    parser.add_argument('--k', type=int, default=20, dest='k_val',
                        help='Hyperparamter K')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    calculate_similar_movies(args.outputfile,
                             model_name=args.model,
                             min_rating=args.min_rating, variant=args.variant, K = args.k_val)




                             