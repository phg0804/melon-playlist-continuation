# Melon Playlist Continuation
카카오 아레나 Melon Playlist Continuation 대회에 참여한 내용을 정리한 리파지토리입니다. 

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

### 1.1. 학습 데이터 생성 (TODO)
train + val 합치는거 TODO

```bash
$> python TODO.py
``` 
생성 결과 파일은 ./res/train_val.json 입니다. 
이 파일은 res/train.json과 res/val.json을 모델의 인풋으로 넣기 위해 하나의 json파일로 합친 것입니다.

### 1.2. 모델 학습 (이것저것 TODO)

```bash
$> python train.py train \
   --train_fpath=res/train_val.json \ 
   --test_fpath=res/test.json
``` 

| model | 설명 | 용량 |
|---|:---:|---:|
| model1.pkl | myALS | 120MB |
| model2.pkl | BM25 | 12MB |
| model3.pkl | BM25 | 12MB |
| model4.pkl | w2v | TODOMB | 

train.py의 최종 결과물은 model1 ~ model4 파일입니다.

### 1.3. 평가 데이터 생성 
inference.py을 실행하면 res/results.json 결과 파일이 생성됩니다.

```bash
$> python inference.py infer \
   --test_fpath=res/test.json \
   --result_fpath=res/results.json
``` 
TODO
## 2. 알고리즘

### 2.1. ALS

### 2.1.1 myALS
### 2.1.2 ALS

### 2.2. BM25, Cosine

### 2.3 w2v
