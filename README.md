# Melon Playlist Continuation
카카오 아레나 Melon Playlist Continuation 대회에 참여한 내용을 정리한 repository입니다. 

## 필요 모듈
- implicit
- gensim
- numpy
- pandas
- fire

## 개발 환경
Colab PRO 
- GPU 16280MiB 
- RAM 25.51GB
- Linux-4.19.104+-x86_64-with-Ubuntu-18.04-bionic
- Python 3.6.9

## 실행
### Step 1. 데이터 다운로드
아레나 홈페이지 https://arena.kakao.com/c/7/data 에 제공되는 파일을 
res 디렉토리에 다운로드 받습니다. 

```bash
├── res
   ├── train.json
   ├── val.json
   ├── test.json
   └── song_meta.json
``` 


### Step 2. 모델 학습
#### Step 2-1. Baseline (genre_most_popular) 실행
```bash
$> python genre_most_popular.py run \
 	   --song_meta_fname=res/song_meta.json \
 	   --train_fname=res/train.json \
 	   --question_fname=res/test.json 
```
결과는 `arena_data/results/results.json`에 저장됩니다.


#### Step 2-2. 모델 학습
```bash
$> python train.py train \
   --train_fname=res/train_val.json \ 
   --test_fname=res/test.json
```
위 command를 실행하면 다음과 같은 파일들이 생성됩니다.

~~추가부탁~~

| model | 설명 | 용량 |
|---|:---:|---:|
| coo.txt | preprocessing | **TODO** |
| model1.pkl | myALS | 120MB |
| model2.pkl | BM25 | 12MB |
| model3.pkl | BM25 | 12MB |
| w2v_tags.pkl | w2v | 40MB |
| w2v_model.pkl | w2v | 210MB |
| model_tag_w1.pkl | w2v | **TODO** |
| model_tag_w2.pkl | w2v | **TODO** |
| model_tag_w3.pkl | w2v | **TODO** |
| model_tag1.txt | cosine | **TODO** |
| model_tag2.txt | bm25 | **TODO** |
| model_tag3.txt | als | **TODO** |
| model_tag4.txt | cosine | **TODO** |
| model_tag5.txt | bm25 | **TODO** |
| model_tag6.txt | als | **TODO** |
| model_tag7.txt | cosine | **TODO** |
| model_tag8.txt | bm25 | **TODO** |
| w2v_results.pkl | 20MB | 

생성되는 파일들의 디렉토리는 다음과 같습니다.

```bash
├── melon-playlist-continuation
   ├── coo.txt
   ├── model1.pkl
   ├── model2.pkl
   ├── model3.pkl
   ├── w2v_model.pkl
   ├── model_tag_w1.pkl
   ├── model_tag_w2.pkl
   ├── model_tag_w3.pkl
   ├── model_tag1.txt
   ├── model_tag2.txt
   ├── model_tag3.txt
   ├── model_tag4.txt
   ├── model_tag5.txt
   ├── model_tag6.txt
   ├── model_tag7.txt
   ├── model_tag8.txt
   └── arena_data
      └── results
         ├── results.json
         └── w2v_results.json
``` 


### Step 3. 평가 데이터 생성 
inference.py을 실행하면 `res/results.json` 결과 파일이 생성됩니다.

```bash
$> python inference.py infer \
   --test_fname=res/test.json \
   --result_fname=res/results.json
``` 


## 2. 알고리즘

우리는 주로 matrix factorization 기반인 ALS와 neighborhood-base learning인 bm25와 cosine을 사용하였고, 자연언어처리 모델인 gensim의 word2vec을 사용하였습니다.
ALS와 bm25, cosine은 implicit의 library에 있는 것을 사용하였고, myals는 als를 개조하여 사용하였습니다.

### 1. ALS, myALS

- myALS  
   다음의 모델들에 대하여 song을 예측하였습니다.
   1. model1.pkl (K=1024, myALS)

- ALS  
   다음의 모델들에 대하여 tag를 예측하였습니다.
   1. model_tag3.txt (als)
   2. model_tag6.txt (als)



### 2.2. BM25, Cosine

다음의 모델들에 대하여 song을 예측하였습니다.
1. model2.pkl (BM25)
2. model3.pkl (BM25)

다음의 모델들에 대하여 태그를 예측하였습니다.
1. model_tag1.txt (cosine)
2. model_tag4.txt (cosine)
3. model_tag7.txt (cosine)
4. model_tag2.txt (BM25)
5. model_tag5.txt (BM25)
6. model_tag8.txt (BM25)

### 2.3 Word2Vec

Word2Vec는 포럼에 올라와있는 코드(https://arena.kakao.com/forum/topics/232 )를 바탕으로 되어있습니다.  
저희 팀의 코드는 word2vec.py에 적혀있습니다.


다음과 같은 모델들에 대하여 song을 예측하였습니다. word2vec model을 만드는데는 20분 정도 소요됩니다.
1. w2v_model.pkl(min_count=1, size=30, window=100, sg=1)

학습한 모델을 바탕으로 가중치들과 neighborhood의 개수등을 조금씩 조절하여 ensemble할 수 있는 데이터들을 몇 가지 만들었습니다.  
만든 데이터들은 다음과 같습니다.
1. arena_data/results/w2v_results.json
2. model_tag_w1.pkl
3. model_tag_w2.pkl
4. model_tag_w3.pkl

### 2.4. Ensemble

위에서 만든 데이터를 기반으로, song과 tag 각각 따로 앙상블을 실시하여 결과를 도출했습니다.
| feature | model |
|---|------|
| song | myALS, BM25(K=2), BM25(K=6), Word2Vec |
| tag  | ALS, cosine, BM25, Word2Vec |
