import fire
import os
import pandas as pd
import pickle
import time
from tqdm import tqdm
from word2vec import Word2VecTrainer

from arena_util import load_json, remove_seen, write_json
from myimplicit import calculate_similar_playlists

class ArenaTrainer:
    def _make_coo(self):
        f = open('./coo.txt', 'w')
        
        song_meta = load_json('./res/song_meta.json')
        num_song = len(song_meta)
        num_tag = len(self.tag2id.keys())
        del song_meta
        N = sum(self.data['songs'].apply(len)) + sum(self.data['tags'].apply(len))
        maxrow = self.data['id'].max()

        f.write("%d %d %d\n" % (maxrow + 1, num_song + num_tag, N))
        
        for i, q in self.train.iterrows():
            for song in q['songs']:
                f.write("%d %d %d %d\n" % (q['id'], song, 1, 0))
            for tag in q['tags']:
                f.write("%d %d %d %d\n" % (q['id'], self.tag2id[tag] + num_song, 1, 0))
        f.close()
        f = open("./coo.txt", 'a')
        for i, q in self.test.iterrows():
            for song in q['songs']:
                f.write("%d %d %d %d\n" % (q['id'], song, 1, 1))
            for tag in q['tags']:
                f.write("%d %d %d %d\n" % (q['id'], self.tag2id[tag] + num_song, 1, 1))
        f.close()
        
    def _get_ans(self):
        similar_playlists = {}
        ans = []

        f = open("./similar-playlist.txt", 'r')
        while True:
            line = f.readline()
            if not line:
                break
            target_plylst, similar_plylst = map(int, line.split())
            if target_plylst==similar_plylst:
                continue
            if target_plylst in similar_playlists.keys():
                similar_playlists[target_plylst].append(similar_plylst)
            else:
                similar_playlists[target_plylst] = [similar_plylst]
        f.close()

        if os.path.isfile("./similar-playlist.txt"):
           os.remove("./similar-playlist.txt")

        for i, q in tqdm(self.test.iterrows()):
            if q['id'] in similar_playlists.keys():
                most_id = similar_playlists[q['id']]
                get_song = []
                get_tag = []
                for id in most_id:
                    get_song += self.song_dict[int(id)]
                    get_tag += self.tag_dict[int(id)]
                get_song = list(pd.value_counts(get_song)[:200].index)
                get_tag = list(pd.value_counts(get_tag)[:20].index)
                ans.append({
                    'id': q['id'],
                    'songs': remove_seen(q['songs'], get_song)[:200],
                    'tags': remove_seen(q['tags'], get_tag)[:20]
                })
            else:
                ans.append({
                  'id': self.w2v_results.loc[i]['id'],
                  'songs': self.w2v_results.loc[i]['songs'],
                  'tags': self.w2v_results.loc[i]['tags']
                })

        return ans

    def _get_ans_myals(self):
        with open("./predicted_songs.pkl", 'rb') as f:
            songs_list = pickle.load(f)
        with open("./predicted_tags.pkl", 'rb') as f:
            tags_list = pickle.load(f)
        if os.path.isfile("./predicted_songs.pkl"):
           os.remove("./predicted_songs.pkl")
        if os.path.isfile("./predicted_tags.pkl"):
           os.remove("./predicted_tags.pkl")

        cnt = 0
        ans = []

        for i, q in tqdm(self.test.iterrows()):
            if q['songs'] != [] or q['tags'] != []:
                ans.append({
                    'id': q['id'],
                    'songs': remove_seen(q['songs'], songs_list[cnt])[:200],
                    'tags': remove_seen(q['tags'], [self.id2tag[_] for _ in tags_list[cnt]])[:20],
                })
                cnt += 1
            else:
                ans.append({
                  'id': self.w2v_results.loc[i]['id'],
                  'songs': self.w2v_results.loc[i]['songs'],
                  'tags': self.w2v_results.loc[i]['tags'],
                })

        return ans

    def _save_models(self, *args):
       
        for i, answers in enumerate(args):
            with open("./model_song" + str(i + 1) + ".pkl", 'wb') as f:
                pickle.dump(answers, f)        
            
    def _train(self, train_fname, test_fname):
        self.train = pd.read_json(train_fname, encoding='UTF-8')
        self.test = pd.read_json(test_fname, encoding='UTF-8')
        self.data = pd.concat([self.train, self.test])
        
        myw2v = Word2VecTrainer(train_fname=train_fname, 
                                test_fname=test_fname, 
                                most_results_fname="./arena_data/results/results.json")
        myw2v.run(topn=80, song_weight=1, tag_weight=2, 
                  title_weight=4, tag_filename = 'model_tag_w1.pkl', save_model=True)
        tag_set = set([])

        for i, q in self.train.iterrows():
          for s in q['tags']:
            tag_set.add(s)
        for i, q in self.test.iterrows():
          for s in q['tags']:
            tag_set.add(s)
        self.tag2id = {x : i for i, x in enumerate(list(tag_set))}
        self.id2tag = {i : x for i, x in enumerate(list(tag_set))}

        self._make_coo()
        
        self.w2v_results = pd.read_json("./arena_data/results/w2v_results.json", encoding='UTF-8')
        self.data = self.data.set_index('id')
        self.song_dict = self.data['songs'].to_dict()
        self.tag_dict = self.data['tags'].to_dict()
        with open("tag_dict.pkl", "wb") as f :
            pickle.dump(self.tag_dict, f)

        calculate_similar_playlists(model_name="myals", factors=1024, test_fname=test_fname)
        answers1 = self._get_ans_myals()        
        calculate_similar_playlists(model_name="bm25", B=0.75, K=2)
        answers2 = self._get_ans()
        calculate_similar_playlists(model_name="bm25", B=0.75, K=6)
        answers3 = self._get_ans()
        calculate_similar_playlists(model_name="cosine", K=6)
        answers4 = self._get_ans()
        self._save_models(answers1, answers2, answers3, answers4)

        myw2v.run(topn=100, with_w2v_model=True, w2v_model='w2v_model.pkl', save_model=True, song_weight=1, tag_weight=1, 
                  title_weight=2, tag_filename = 'model_tag_w2.pkl', write_results = False)
        myw2v.run(topn=150, with_w2v_model=True, w2v_model='w2v_model.pkl', save_model=True, song_weight=1, tag_weight=1, 
                  title_weight=2, tag_filename = 'model_tag_w3.pkl', write_results = False)
        calculate_similar_playlists("model_tag1.txt", model_name='cosine', K=200)
        calculate_similar_playlists("model_tag2.txt", model_name='bm25', B=0.2, K=200)
        calculate_similar_playlists("model_tag3.txt", model_name='als', factors=96)
        calculate_similar_playlists("model_tag4.txt", model_name='cosine', K=250)
        calculate_similar_playlists("model_tag5.txt", model_name='bm25', B=0.2, K=250)
        calculate_similar_playlists("model_tag6.txt", model_name='als', factors=64)
        calculate_similar_playlists("model_tag7.txt", model_name='cosine', K=150)
        calculate_similar_playlists("model_tag8.txt", model_name='bm25', B=0.2, K=150)

    def train(self, train_fname, test_fname, val_fname=""):
        try:
            if not val_fname:
                self._train(train_fname, test_fname)
            else:
                new_train_fname = "./res/train_val.json"
                train = load_json(train_fname)
                val = load_json(val_fname)
                write_json(train + val, "./../res/train_val.json")
                self._train(new_train_fname, test_fname)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    # only for execution using colab
    from google.colab import drive
    import subprocess
    drive.mount('/content/gdrive/')
    subprocess.call("cd /content/gdrive/My\ Drive/melon-playlist-continuation", shell=True)
    subprocess.call("pwd", shell=True)

    fire.Fire(ArenaTrainer)
