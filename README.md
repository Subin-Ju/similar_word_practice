# 유사한 단어 찾기 게임
> 데이터셋에서 유사한 단어를 찾아 `A:B = C:D` 를 만족하는 D를 찾아보는 게임입니다.

<br>

## 활용한 데이터
> 데이터 출처: https://data.seoul.go.kr/dataList/OA-22397/F/1/datasetView.do  
> - 재난 이슈 유형별 말뭉치(corpus)

<br>

## 데이터 전처리
```
# 데이터 전처리
from konlpy.tag import Okt
from tqdm import tqdm


# 한국어 불용어 처리
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='UTF-8-sig') as f:
        stopwords = [line.strip() for line in f]
    return stopwords

ko_stopwords = load_stopwords('ko_stopwords.txt')

okt = Okt()
stopwords_ted =  ["은", "는", "이", "가", "을", "를", "과", "와", "들", "도", "하다", "이다", "께서", "에서","에게", "의", "이라고", "이라고",
                  "으로", "로", "에서", "에게서", "으로부터", "로부터", "으로써", "로써"
                "부터", "까지", "에", "나", "너", "그", "걔", "얘"]

preprocessed_text = []

# tqdm 활용
for text in tqdm(game_df['text']):
    tokens = okt.morphs(text, stem=True)
    tokens = [token for token in tokens if token not in ko_stopwords]
    tokens = [token for token in tokens if token not in stopwords_ted]
    preprocessed_text.append(tokens)

preprocessed_text
```

<br>

## 모델 학습 결과
```
fasttext_model.wv.most_similar('방사능')

[('방사성', 0.9856238961219788),
 ('방사', 0.9842268824577332),
 ('방사청', 0.976813793182373),
 ('능', 0.9678596258163452),
 ('방사선', 0.966058075428009),
 ('상방사', 0.9332483410835266),
 ('세슘', 0.9089751839637756),
 ('탐사선', 0.878200888633728),
 ('오염수', 0.8547028303146362),
 ('라돈', 0.8510088920593262)]

```

## 실행 결과
```
6700억:9300억 = 6700만:9300만
8300만
C와 D 유사도: 0.95
당신은 유추의 천재🪄
```
- 학습 성능 개선이 필요..
