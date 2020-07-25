import pandas as pd
import random
import pickle
import time
from tqdm import tqdm
from arena_util import remove_seen, write_json, load_json

from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, \
                                         remove_stopwords, stem_text, \
                                         strip_multiple_whitespaces

def date2int(date_str):
    return int(date_str[:4]) * 10000 + int(date_str[5:7]) * 100 + int(date_str[8:10])

def shuffle_list(lst, seed=0):
    random.seed(seed)
    random.shuffle(lst)
    return lst

def tostring(lst):
    return [str(x) for x in lst]

    
class Word2VecTrainer:
    def __init__(
      self, 
      train_fname="./res/train.json",
      test_fname = "./res/test.json",
      most_results_fname="./arena_data/results/result.json"
    ):
        self.train = pd.read_json(train_fname, encoding='UTF-8')
        self.test = pd.read_json(test_fname, encoding='UTF-8')
        self.song_meta = pd.read_json("./res/song_meta.json", encoding='UTF-8')
        self.most_results = pd.read_json(most_results_fname, encoding='UTF-8')
        
        self.min_count = 1
        self.size = 30
        self.windows = 100
        self.sg = 1
        self.custom_filters = [remove_stopwords, stem_text, 
                               strip_punctuation, strip_multiple_whitespaces]
    
    def _get_dic(self, train, test, song_meta):
        tqdm.pandas()
        data = pd.concat([train, test])
        data = data.set_index('id')

        self.song_dict = data['songs'].apply(lambda x : shuffle_list(x, 123)).apply(lambda x : tostring(x)).to_dict()
        self.tag_dict = data['tags'].apply(lambda x : shuffle_list(x, 123)).to_dict()
        
        data = data.reset_index()
        self.total = data.progress_apply(lambda x : self.song_dict[x['id']] + self.tag_dict[x['id']] + preprocess_string(x['plylst_title'], self.custom_filters), axis = 1)

    def _get_w2v(self, save_model=True):
        print("Started traing Embedding... This might take about 20mins")
        time1 = time.time()
        self.w2v_model = Word2Vec(self.total, min_count = self.min_count, size = self.size, window = self.windows, sg = self.sg)
        time2 = time.time()
        print("Embedding completed. Took " + "{:.2f}".format((time2 - time1)/60) + "min ")
        
        if save_model:
            with open('w2v_model.pkl', 'wb') as f:
                pickle.dump(self.w2v_model, f)
        
    def _playlist2vec(self, song_weight = 1, tag_weight = 1, title_weight = 1):
        self.p2v_model = WordEmbeddingsKeyedVectors(self.size)
        id = []   
        vec = []
        for index, q in tqdm(pd.concat([self.train, self.test]).iterrows()):
            tmp_vec = 0
            for song in self.song_dict[q['id']]:
                try:
                    tmp_vec += song_weight * self.w2v_model.wv.get_vector(song) / len(self.song_dict[q['id']])
                except KeyError:
                    pass
            for tag in self.tag_dict[q['id']]:
                try:
                    tmp_vec += tag_weight * self.w2v_model.wv.get_vector(tag) / len(self.tag_dict[q['id']])
                except KeyError:
                    pass
            for title_word in preprocess_string(q['plylst_title'], self.custom_filters):
                try:
                    tmp_vec += title_weight * self.w2v_model.wv.get_vector(title_word) / len(preprocess_string(q['plylst_title'], self.custom_filters))
                except KeyError:
                    pass

            if type(tmp_vec) != int:
                id.append(str(q['id']))  
                vec.append(tmp_vec)

        self.p2v_model.add(id, vec)
        
    def _get_results(self, topn = 80):
        print("extracting results")
        answers = []
        tags = []
        for index, q in tqdm(self.test.iterrows()):
            try:
                most_id = [x[0] for x in self.p2v_model.most_similar(str(q['id']), topn=topn)]
                get_song = []
                get_tag = []
                for id in most_id:
                    get_song += self.song_dict[int(id)]
                    get_tag += self.tag_dict[int(id)]
                
                get_song = list(pd.value_counts(get_song)[:200].index)
                get_song = [int(x) for x in get_song]
                
                tags.append(get_tag)
                
                updt_date = date2int(q['updt_date'])
                song_date = self.song_meta.loc[get_song, 'issue_date']
                get_song = pd.Series(get_song)[[x <= updt_date for x in song_date]] 
                
                get_tag = list(pd.value_counts(get_tag)[:20].index)
                answers.append({
                    "id": q["id"],
                    "songs": remove_seen(q["songs"], get_song)[:100],
                    "tags": remove_seen(q["tags"], get_tag)[:10],
                })
            except KeyError:
                tags.append([])
                answers.append({
                  "id": self.most_results.loc[index]["id"],
                  "songs": self.most_results.loc[index]['songs'],
                  "tags": self.most_results.loc[index]["tags"],
                })
                
        # check and update answer
        for n, q in enumerate(answers):
            if len(q['songs'])!=100:
                answers[n]['songs'] += remove_seen(q['songs'], self.most_results.loc[n]['songs'])[:100-len(q['songs'])]
            if len(q['tags'])!=10:
                answers[n]['tags'] += remove_seen(q['tags'], self.most_results.loc[n]['tags'])[:10-len(q['tags'])]  
        self.answers = answers
        self.tags = tags
        
        with open("w2v_tags.pkl", 'wb') as f:
            pickle.dump(self.tags, f)
        print("tags written to w2v_tags.pkl")
        
    def run(self, topn=80, with_w2v_model=False, w2v_model=None, save_model=True, 
            song_weight=1, tag_weight=1, title_weight=1):
        self._get_dic(self.train, self.test, self.song_meta)
        
        if with_w2v_model :
            with open(w2v_model, 'rb') as f:
                self.w2v_model = pickle.load(f)
            print("Traning Started with " + w2v_model)
        else:    
            self._get_w2v(save_model)
            
        self._playlist2vec(song_weight, tag_weight, title_weight) self._get_results(topn)
        write_json(self.answers, "./results/w2v_results.json")

