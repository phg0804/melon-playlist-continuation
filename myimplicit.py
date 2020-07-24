from __future__ import print_function

import argparse
import codecs
import logging
import numpy as np
import time
import tqdm
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         bm25_weight)

import arena_data as au
from myals import MyAlternatingLeastSquares


log = logging.getLogger("implicit")


def calculate_similar_playlists(output_filename='similar-playlist.txt',
                                model_name="als", 
                                K=2):
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
    model = AlternatingLeastSquares(factors=K)

    # lets weight these models by bm25weight.
    log.debug("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()

  elif model_name == "myals":
    model = MyAlternatingLeastSquares(factors=K)

    # lets weight these models by bm25weight.
    log.debug("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()
  
  elif model_name == "cosine":
    model = CosineRecommender()

  elif model_name == "bm25":
    model = BM25Recommender(B=0.75, K=K)

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
            o.write("%s %s %s\n" % (playid, other, score))
        progress.update(1)
    
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Generates related playlists / songs and tags ",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument('--output', type=str, default='similar-playlist.txt',
                      dest='outputfile', help='output file name')
  parser.add_argument('--model', type=str, default='bm25',
                      dest='model', 
                      help='model to calculate (als, myals, cosine, bm25)')
  parser.add_argument('--K', type=int, default=2, dest='K',
                      help='Parameter for als, myals and bm25')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG)

  calculate_similar_playlists(args.outputfile,
                              model_name=args.model,
                              K=args.K)