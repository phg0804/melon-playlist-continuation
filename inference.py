import fire
import pandas as pd
import pickle

from arena_util import load_json, write_json, remove_seen
from google.colab import drive


def date2int(date_str):
    return int(date_str[:4]) * 10000 + int(date_str[5:7]) * 100 + int(date_str[8:10])

class ArenaInferrer:
  def filter_future_songs(self, ans, df):
    for i, q in enumerate(ans):
      new_songs_list = []
      plylst_date = date2int(df.iloc[i]['updt_date'])
      for song in q['songs']:
        song_date = int(self.song_meta[song]['issue_date'])
        if song_date <= plylst_date:
          new_songs_list.append(song)
      q['songs'] = new_songs_list

  def ensemble(self, answers_list, song_param1, song_param2):
    answers = []
    similar_playlist_list = []
    fname_list = ["model_tag1.txt", "model_tag2.txt", "model_tag3.txt", "model_tag4.txt", 
                "model_tag5.txt", "model_tag6.txt", "model_tag7.txt", "model_tag8.txt"]
    with open('model_tag_w1.pkl', 'rb') as f: 
      tag_ld_w2v_80 = pickle.load(f)
    with open('model_tag_w2.pkl', 'rb') as f: 
      tag_ld_w2v_100 = pickle.load(f)
    with open('model_tag_w3.pkl', 'rb') as f: 
      tag_ld_w2v_150 = pickle.load(f)
    
    for fname in fname_list :
      similar_playlists = {}
      with open(fname, 'r') as f :
        while True:
            line = f.readline()
            if not line:
                break
            target_plylst, similar_plylst = map(int, line.split())
            if target_plylst == similar_plylst:
                continue
            if target_plylst in similar_playlists.keys():
                similar_playlists[target_plylst].append(similar_plylst)
            else:
                similar_playlists[target_plylst] = [similar_plylst]
      similar_playlist_list.append(similar_palylits)
    
    for i in range(len(answers_list[0])):
      tmp = {}
      tmp_song = {}
      tmp_tag = {}
        
      for k, ans in enumerate(answers_list):
        for n, song in enumerate(ans[i]['songs']):
          if song in tmp_song.keys():
            tmp_song[song] += 1 / (n + song_param1[k])
          else:
            tmp_song[song] = 1 / (n + song_param2[k])    
        sorted_songs = sorted(tmp_song.items(), reverse=True, key=lambda _: _[1])
        sorted_songs = [k for (k, v) in sorted_songs]
      tmp['id'] = answers_list[0][i]['id']
      tmp['songs'] = sorted_songs[:200]
      tmp['tags'] = [str(_) for _ in range(10)]
      answers.append(tmp)
    
    with open("tag_dict.pkl", "rb") as f:
        tag_dict = pickle.load(f)
    
    for i, q in self.test.iterrows():
        
      try:
        get_song = []
        get_tag = []

        for s_l in similar_playlist_list :
          most_id = s_l[q['id']]
          for id in most_id:
              get_tag += tag_dict[int(id)]
        get_tag += tag_ld_w2v_100[i]
        get_tag += tag_ld_w2v_100[i]
        get_tag += tag_ld_w2v_100[i]
        get_tag += tag_ld_w2v_80[i]
        get_tag += tag_ld_w2v_80[i]
        get_tag += tag_ld_w2v_80[i]
        get_tag += tag_ld_w2v_80[i]
        get_tag += tag_ld_w2v_150[i]
        get_tag += tag_ld_w2v_150[i]
        
        get_tag = list(pd.value_counts(get_tag).index)
        get_tag = remove_seen(q["tags"], get_tag)

        if len(get_tag)!= 10 :
          get_tag += remove_seen(get_tag, self.w2v_results[i]['tags'])[:10-len(get_tag)]

        answers[i] = {
          "id": q["id"],
          "songs": remove_seen(q["songs"], answers[i]['songs'])[:100],
          "tags": remove_seen(q["tags"], get_tag)[:10]
        }
    
      except KeyError:
        answers[i] = {
          "id": q["id"],
          "songs": remove_seen(q["songs"], answers[i]['songs'])[:100],
          "tags": remove_seen(q["tags"], self.w2v_results[i]['tags'])[:10]
        }
       
    for i, q in enumerate(answers):
      if len(q['songs']) < 100:
        answers[i]['songs'] += remove_seen(q['songs'], self.w2v_results[i]['songs'])[:100-len(q['songs'])]
      if len(q['tags']) < 10:
        answers[i]['tags'] += remove_seen(q['tags'], self.w2v_results[i]['tags'])[:10-len(q['tags'])]
    for i, q in self.test.iterrows():
      if len(q['songs']) == 0:
        answers[i]['songs'] = self.w2v_results[i]['songs'][:100]

    return answers
  
  def _infer(self, test_fname, result_fname):
    # Load models
    with open("./model_song1.pkl", 'rb') as f:
      answers1 = pickle.load(f)
    with open("./model_song2.pkl", 'rb') as f:
      answers2 = pickle.load(f)
    with open("./model_song3.pkl", 'rb') as f:
      answers3 = pickle.load(f)
    
    self.test = pd.read_json(test_fname, encoding='UTF-8')
    self.w2v_results = load_json("./arena_data/results/w2v_results.json")
    self.song_meta = load_json("./res/song_meta.json")
    self.filter_future_songs(answers1, self.test)
    self.filter_future_songs(answers2, self.test)
    self.filter_future_songs(answers3, self.test)
    
    answers = self.ensemble([answers1, answers2, answers3], 
                            [15, 30, 50], [18, 40, 110])
    write_json(answers, result_fname)

  def infer(self, test_fname, result_fname):
    try:
      self._infer(test_fname, result_fname)
      print("The result is successfully generated")
    except Exception as e:
      print(e)  

if __name__ == "__main__":
  # only for execution using colab
  from google.colab import drive
  import subprocess
  drive.mount('/content/gdrive/')
  subprocess.call("cd /content/gdrive/My\ Drive/melon-playlist-continuation", shell=True)
  subprocess.call("pwd", shell=True)

  fire.Fire(ArenaInferrer)

