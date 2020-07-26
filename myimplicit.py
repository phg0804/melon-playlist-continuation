from __future__ import print_function

import argparse
import codecs
import logging
import numpy as np
import os
import time
import tqdm
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         bm25_weight)

import arena_util as au
from myals import MyAlternatingLeastSquares


log = logging.getLogger("implicit")


def calculate_similar_playlists(output_filename="similar-playlist.txt",
                                model_name="als", 
                                test_fname="./res/test.json",
                                K=20, B=0.75, factors=50):
  # read in the input data file
  start = time.time()
  
  rows = []
  cols = []
  scores = []
  mask = []
  with open("./coo.txt", 'r') as f:
    num_row, num_col, N = map(int, f.readline().split())
    mask = [False] * num_row
    for _ in range(N):
      r, c, s, m = map(int, f.readline().split())
      if m == 1: # 0 : train / 1 : val, test
        mask[r] = True
      rows.append(r)
      cols.append(c)
      scores.append(float(s))
  ratings = csr_matrix((np.array(scores, dtype=np.float32), 
                        (np.array(rows), np.array(cols))), 
                        shape = (num_row, num_col))
  
  ratings.data = np.ones(len(ratings.data))
  log.info("read data file in %s", time.time() - start)

  # generate a recommender model based off the input params
  if model_name == "als":
    model = AlternatingLeastSquares(factors=factors)

    # lets weight these models by bm25weight.
    log.debug("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()

  elif model_name == "myals":
    song_meta = au.load_json('./res/song_meta.json')
    num_song = len(song_meta)
    del song_meta
    model = MyAlternatingLeastSquares(num_song=num_song, 
                                      num_tag=num_col-num_song, 
                                      factors=facotrs,
                                      test_fname=test_fname)

    # lets weight these models by bm25weight.
    log.debug("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()
  
  elif model_name == "cosine":
    model = CosineRecommender(K=K)

  elif model_name == "bm25":
    model = BM25Recommender(B=B, K=K)

  else:
    raise NotImplementedError("TODO: model %s" % model_name)

  # train the model
  log.debug("training model %s", model_name)
  start = time.time()
  model.fit(ratings)
  log.debug("trained model '%s' in %s", model_name, time.time() - start)
  if model_name == "myals":
    return
  log.debug("calculating top playlists")

  user_count = np.ediff1d(ratings.indptr)
  to_generate = np.arange(num_row)
  
  log.debug("calculating similar playlists")
  with tqdm.tqdm(total=len(to_generate)) as progress:
    with codecs.open(output_filename, "w", "utf8") as o:
      for playid in to_generate:
        if ratings.indptr[playid] != ratings.indptr[playid+1] and mask[playid]:
          for other, score in model.similar_items(playid, 201):
            o.write("%s %s\n" % (playid, other))
        progress.update(1)
    
