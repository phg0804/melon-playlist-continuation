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

## 1. 실행
### 1.0. 데이터 다운로드
아레나 홈페이지 https://arena.kakao.com/c/7/data 에 제공되는 파일을 
res 디렉토리에 다운로드 받습니다. 

```bash
├── res
   ├── train.json
   ├── val.json
   ├── test.json
   ├── song_meta.json
   └── genre_gn_all.json
``` 


### 1.1. 모델 학습 (이것저것 TODO)

```bash
$> python train.py train \
   --train_fname=res/train_val.json \ 
   --test_fname=res/test.json
```
위 command를 실행하면 다음과 같은 파일들이 생성됩니다.

| model | 설명 | 용량 |
|---|:---:|---:|
| model1.pkl | myALS | 120MB |
| model2.pkl | BM25 | 12MB |
| model3.pkl | BM25 | 12MB |
| w2v_results.pkl | 20MB | 
| w2v_tags.pkl | w2v | 40MB |
| w2v_model.pkl | w2v | 210MB | 
~~앵버가 추가~~

생성되는 파일들의 디렉토리는 다음과 같습니다.

```bash
├── melon-playlist-continuation
   ├── coo.txt
   ├── model1.pkl
   ├── model2.pkl
   ├── model3.pkl
   ├── w2v_model.pkl
   ├── w2v_tags.pkl
   └── arena_data
      └── results
         ├── results.json
         └── w2v_results.json
``` 

~~앵버형이 만든 tag 관련 파일들이 어떻게 적히는가 보아야 함~~


### 1.2. 평가 데이터 생성 
inference.py을 실행하면 res/results.json 결과 파일이 생성됩니다.

```bash
$> python inference.py infer \
   --test_fname=res/test.json \
   --result_fname=res/results.json
``` 
위의 커맨드를 실행하면

```bash
melon-playlist-continuation/res/results.json
``` 
에 결과값이 생성됩니다.

## 2. 알고리즘

우리는 주로 matrix factorization기반인 ALS와 neighborhood bas learning인 bm25와 cosine을 사용하였고, 자연언어처리 모델인 gensim의 word2vec을 사용하였습니다.
ALS와 bm25, cosine은 implicit의 library에 있는 것을 사용하였고, myals는 als를 약간 개조하여 사용하였습니다.

### 2.1. ALS, myALS

다음과 같은 모델들에 대하여 song을 예측하였습니다.
1. model1.pkl (K = 1024, myALS)

### 2.2. BM25, Cosine

다음과 같은 모델들에 대하여 song을 예측하였습니다.
1. model2.pkl (K = 2, BM25)
2. model3.pkl (K = 6, BM25)

### 2.3 w2v

w2v는 포럼에 올라와있는 코드를 참고하여 살짝 수정하였습니다. 
저희 팀의 코드는 word2vec.py에 적혀있습니다.
Original code는 다음과 같은 주소에 있습니다.
https://arena.kakao.com/forum/topics/232

다음과 같은 모델들에 대하여 song을 예측하였습니다.
1. w2v_model.pkl( embedding space dim = 30, windows = 100)

학습한 모델을 바탕으로 가중치들과 neighborhood의 개수등을 조금씩 조절하여 ensemble할 수 있는 데이터들을 몇 가지 만들었습니다.
데이터들의 추출방식은 다음과 같습니다.
1. arena_data/results/w2v_results.json
2. ~~앵버가 만든 태그들 넣기~~


### 2.4. Ensemble

위에서 만든 데이터를 기반으로, song과 tag 각각 따로 앙상블을 실시하였습니다.
