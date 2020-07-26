# Melon Playlist Continuation
카카오 아레나 Melon Playlist Continuation 대회에 참여한 내용을 정리한 repository입니다. 

## 필요 모듈
- fire
- gensim
- implicit
- numpy
- pandas
- tqdm

## 개발 환경
Colab PRO 
- GPU 16280MiB 
- RAM 25.51GB
- Linux-4.19.104+-x86_64-with-Ubuntu-18.04-bionic
- Python 3.6.9

**myALS 실행을 위하여 GPU가 반드시 필요합니다.** (CPU 버전 미구현)

## 실행
### Step 1. 데이터 다운로드
[아레나 홈페이지](https://arena.kakao.com/c/7/data) 에 제공되는 파일을 
res 디렉토리에 다운로드 받습니다. 

```bash
├── melon-playlist-continuation
   ├── res
      ├── train.json
      ├── val.json
      ├── test.json
      └── song_meta.json
``` 


### Step 2. 모델 학습
모델 [다운로드](https://drive.google.com/drive/folders/1AKdPXtyAl8nFA0i325pHdtNTDfxxYfmQ?usp=sharing)
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
   --train_fname=res/train.json \ 
   --test_fname=res/test.json \
   --val_fname=res/val.json
```
위 command를 실행하면 `coo.txt`(ALS 알고리즘을 위한 데이터)와 `tag_dict.pkl`(tag를 id와 매핑한 딕셔너리의 pickle 파일)과 함께
다음과 같은 모델들이 생성됩니다.

| model | 설명 | 용량 |
|---|:---:|---:|
| model_song1.pkl | myALS | 55MB |
| model_song2.pkl | BM25 | 3MB |
| model_song3.pkl | BM25 | 8MB |
| model_song4.pkl | cosine | 5MB |
| model_tag_w1.pkl | w2v | 16MB |
| model_tag_w2.pkl | w2v | 20MB |
| model_tag_w3.pkl | w2v | 27MB |
| model_tag1.txt | cosine | 22MB |
| model_tag2.txt | BM25 | 22MB |
| model_tag3.txt | ALS | 24MB |
| model_tag4.txt | cosine | 22MB |
| model_tag5.txt | BM25 | 22MB |
| model_tag6.txt | ALS | 24MB |
| model_tag7.txt | cosine | 17MB |
| model_tag8.txt | BM25 | 17MB |
| w2v_model.pkl | w2v | 220MB |
| w2v_results.json | w2v | 10MB | 




### Step 3. 평가 데이터 생성 
inference.py을 실행하면 `res/results.json` 결과 파일이 생성됩니다.

```bash
$> python inference.py infer \
   --test_fname=res/test.json \
   --result_fname=/../res/results.json
``` 
결과는 `res/results.json`에 저장됩니다.


생성되는 파일들의 디렉토리는 다음과 같습니다.

```bash
├── melon-playlist-continuation
   ├── coo.txt
   ├── model_song1.pkl
   ├── model_song2.pkl
   ├── model_song3.pkl
   ├── model_song4.pkl
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
   ├── tag_dict.pkl
   ├── w2v_model.pkl
   ├── res
      └── train_val.json
      └── results.json (최종 답안)
   └── arena_data
      └── results
         ├── results.json (baseline 답안)
         └── w2v_results.json   
``` 

## 알고리즘

저희 팀은 주로 matrix factorization 기반인 ALS와 neighborhood-base learning인 BM25와 cosine을 사용하였고, 자연언어처리 모델인 gensim의 word2Vec을 사용하였습니다.
ALS와 BM25, cosine은 implicit library에 있는 것을 사용하였고, myALS는 ALS를 변형하여 사용하였습니다.

### 1. ALS, myALS(Alternating Least Squares)

다음의 모델들에 대하여 song을 예측하였습니다.
1. model_song1.pkl (myALS)

다음의 모델들에 대하여 tag를 예측하였습니다.
1. model_tag3.txt (ALS)
2. model_tag6.txt (ALS)



### 2. BM25, cosine

다음의 모델들에 대하여 song을 예측하였습니다.
1. model_song2.pkl (BM25)
2. model_song3.pkl (BM25)
3. model_song4.pkl (cosine)

다음의 모델들에 대하여 tag를 예측하였습니다.
1. model_tag1.txt (cosine)
2. model_tag4.txt (cosine)
3. model_tag7.txt (cosine)
4. model_tag2.txt (BM25)
5. model_tag5.txt (BM25)
6. model_tag8.txt (BM25)

### 3. Word2Vec

Word2Vec는 카카오 아레나 포럼에 올라와있는 [코드](https://arena.kakao.com/forum/topics/232)를 참고하였습니다.  
저희 팀의 코드는 word2vec.py에 작성하였습니다.


다음과 같은 모델들에 대하여 song을 예측하였습니다. Word2Vec model생성은 20분 정도 소요됩니다.
1. w2v_model.pkl

학습한 모델을 바탕으로 가중치들과 neighborhood의 개수 등을 조절하여 ensemble할 수 있는 데이터들을 생성하였습니다.  
생성된 데이터들은 다음과 같습니다.
1. arena_data/results/w2v_results.json
2. model_tag_w1.pkl
3. model_tag_w2.pkl
4. model_tag_w3.pkl

### 4. Ensemble

위에서 생성한 데이터를 기반으로, song과 tag 각각 따로 앙상블을 실시하여 결과를 도출했습니다.
| feature | model |
|---|------|
| song | myALS, cosine, BM25, Word2Vec |
| tag  | ALS, cosine, BM25, Word2Vec |

tag 예측의 경우, 총 11개의 모델이 각각 주어진 playlist와 가까운 playlist들을 찾고, 그 playlist들에서 가장 자주 등장하는 10개의 tag들을 답안으로 제출하였습니다.

## 대략적 실행시간

위에서 언급된 개발환경에서의 대략적인 실행시간입니다.
| command | time |
|---|------|
| train.py  | 약 90분 |
| inference.py  | 약 3분 |
