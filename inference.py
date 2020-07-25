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

        for n, tag in enumerate(ans[i]['tags']):
          if tag in tmp_tag.keys():
            tmp_tag[tag] += 1 / (n + 15)
          else:
            tmp_tag[tag] = 1 / (n + 15)    
        sorted_tags = sorted(tmp_tag.items(), reverse=True, key=lambda _: _[1])
        sorted_tags = [k for (k, v) in sorted_tags]

      tmp['id'] = answers_list[0][i]['id']
      tmp['songs'] = sorted_songs[:200]
      tmp['tags'] = sorted_tags[:20]
      answers.append(tmp)
    
    for i, q in self.test.iterrows():
      answers[i] = {
        "id": q["id"],
        "songs": remove_seen(q["songs"], answers[i]['songs'])[:100],
        "tags": remove_seen(q["tags"], answers[i]['tags'])[:10]
      }

    for i, q in enumerate(answers):
      if len(q['songs']) < 100:
        answers[i]['songs'] += remove_seen(q['songs'], self.w2v_results.loc[i]['songs'])[:100-len(q['songs'])]
      if len(q['tags']) < 10:
        answers[i]['tags'] += remove_seen(q['tags'], self.w2v_results.loc[i]['tags'])[:10-len(q['tags'])]
    for i, q in self.test.iterrows():
      if len(q['songs']) == 0:
        answers[i]['songs'] = self.w2v_results.loc[i]['songs'][:100]

    return answers
    
  def _infer(self, test_fpath, result_fpath):
    # Load models
    with open("./model1.pkl", 'rb') as f:
      answers1 = pickle.load(f)
    with open("./model2.pkl", 'rb') as f:
      answers2 = pickle.load(f)
    with open("./model3.pkl", 'rb') as f:
      answers3 = pickle.load(f)

    self.test = pd.read_json(test_fpath, encoding='UTF-8')    
    self.w2v_results = pd.read_json('./w2v_local.json', encoding='UTF-8')
    self.song_meta = load_json("./res/song_meta.json")
    self.filter_future_songs(answers1, self.test)
    self.filter_future_songs(answers2, self.test)
    self.filter_future_songs(answers3, self.test)

    answers = self.ensemble([answers1, answers2, answers3], 
                            [15, 30, 50], [18, 40, 110])
    write_json(answers, result_fpath)

  def infer(self, test_fpath, result_fpath):
    try:
      self._infer(test_fpath, result_fpath)
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

