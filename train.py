import fire

import pickle
import subprocess

import os
import time
import pandas as pd
from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from tqdm import tqdm

import myimplicit

#from google.colab import drive
#drive.mount('/content/gdrive/')
#!cd /content/gdrive/My\ Drive/melon-playlist-continuation
#!pwd
subprocess.call('ls', shell=True)

from google.colab import drive

class ArenaTrainer:
        
    def _get_ans(self):
        similar_playlists = {}
        ans = []

        f = open("./similar-playlist.txt", 'r')
        while True:
            line = f.readline()
            if not line:
                break
            target_plylst, similar_plylst, _ = line.split()
            target_plylst = int(target_plylst)
            similar_plylst = int(similar_plylst)
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
                    'tags': remove_seen(q['tags'], get_tag)[:20],
                })
            else:
                ans.append({
                  'id': self.w2v_results.loc[i]['id'],
                  'songs': self.w2v_results.loc[i]['songs'],
                  'tags': self.w2v_results.loc[i]['tags'],
                })     
        return ans

    def _get_ans_myals(self):
        ans = []
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
                    'tags': remove_seen(q['tags'], [self.my_dict_inv[_] for _ in tags_list[cnt]])[:20],
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
            with open("./model" + str(i + 1) + ".pkl", 'wb') as f:
                pickle.dump(answers, f)        
          
    def _train(self, train_fpath, test_fpath):
        self.train = pd.read_json(train_fpath, encoding='UTF-8')
        self.test = pd.read_json(test_fpath, encoding='UTF-8')
        self.data = pd.concat([self.train, self.test])
        self.data = self.data.set_index('id')

        # most_results를 그대로 쓸 것인가?
        # w2v_results 만드는 코드
        self.w2v_results = pd.read_json("./omg2_real.json", encoding='UTF-8')
        self.song_dict = self.data['songs'].to_dict()
        self.tag_dict = self.data['tags'].to_dict()
        # tag_id_map.pkl 만드는 코드
        with open("tag_id_map.pkl", 'rb') as f:
            my_dict = pickle.load(f)
            self.my_dict_inv = {} 
            for k, v in my_dict.items():
                self.my_dict_inv[v] = k    

        myimplicit.calculate_similar_playlists(model_name="myals", iterations=20, K=1024)
        answers1 = self._get_ans_myals()        
        myimplicit.calculate_similar_playlists(model_name="bm25", K=2)
        answers2 = self._get_ans()
        myimplicit.calculate_similar_playlists(model_name="bm25", K=6)
        
        answers3 = self._get_ans()

        self._save_models(answers1, answers2, answers3)

    def train(self, train_fpath, test_fpath):
        try:
            self._train(train_fpath, test_fpath)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    drive.mount('/content/gdrive/')
    subprocess.call("cd /content/gdrive/My\ Drive/melon-playlist-continuation", shell=True)
    subprocess.call("pwd", shell=True)

    fire.Fire(ArenaTrainer)
