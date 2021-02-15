---
layout: post
title: '[ArcFault] Project step02: Data Analysis'
subtitle: 'data analysis'
categories: sideproject
tags: side5
comments: true
---
`Arc-Fault` 연구과제를 위해 개발한 코드입니다.

## Introduction
저번 [포스트](https://geonkimdcu.github.io/sideproject/2021/02/03/SP-ArcFault-2/)에선 데이터 수집 단계에 대해 알아보았습니다. <br>
이번 포스팅은 수집한 데이터를 이용해 분석 후 분류모델 생성까지 해보겠습니다. <br><br>
전체 코드는 제 [Github](https://github.com/GeonKimdcu/SideProject)에 업로드 되어 있습니다.

## Correlation analysis
<br>

먼저 데이터를 load 해줍니다.
```python
# 데이터 load
arc_df = pd.read_csv('Arc_data.csv')
arc_df.head(10)
```

데이터 내 결측치 여부를 확인해줍니다.
```python
# 데이터 내 NA 값 여부 확인
arc_df.isnull().any() # 만약 존재한다면 0으로 대체 혹은, 해당 열을 제외하고 진행
```

이제 본격적으로 상관분석을 해보겠습니다. 먼저 독립변수만 따로 추출하도록 하겠습니다. 
```python
# 독립변수만 따로 추출
features = ['Max', 'Mean', 'Std', 'Full_diff', 'FFT_Mag1', 'FFT_Hz1', 'FFT_Mag2', 'FFT_Hz2',
            'STFT_Mag1', 'STFT_Hz1', 'STFT_Mag2', 'STFT_Hz2', 'STFT_diff']

arc_corr = arc_df[features]
```

그리고 `corr()`함수를 통해 상관계수를 출력해보겠습니다.
```python
# 상관계수 출력(method : 'Pearson')
arc_corr.corr()
```

출력값들을 보시면 대부분 변수들끼리 높은 상관성을 띄고 있는 것을 알 수 있습니다. 이는 나중에 [다중공선성](https://ko.wikipedia.org/wiki/%EB%8B%A4%EC%A4%91%EA%B3%B5%EC%84%A0%EC%84%B1) 문제를 일으키므로 VIF(Variance Inflation Factors, 분산팽창요인) 계수를 활용해 변수 선택법을 진행해보겠습니다. <br><br>

## Feature Select

> ###  VIF 제거 순서
1. VIF 계수가 높은 feature 제거(단, 유사한 feature의 경우 둘 중 1개만 제거)
2. 제거 후 VIF 계수 재출력
3. (1,2)의 과정 반복

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 피처마다의 VIF 계수를 출력합니다.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(arc_corr.values, i) for i in range(arc_corr.shape[1])]
vif["features"] = arc_corr.columns
vif
```
VIF 계수가 높은 변수를 제거해주고, 다시 계수를 출력해보겠습니다.
```python
# VIF 계수 높은 feature 제거

arc_corr = arc_corr.drop(['Std', 'FFT_Mag2', 'FFT_Hz2', 'STFT_Mag2', 'STFT_Hz2', 'STFT_Hz1'], axis = 1)
vif = pd.DataFrame()
vif['VIF Factor']  = [variance_inflation_factor(arc_corr.values, i) for i in range(arc_corr.shape[1])]
vif['features'] = arc_corr.columns
vif
```
![arcpost16](https://user-images.githubusercontent.com/48666867/107337570-e4a1a100-6afd-11eb-9075-2de47a0cc62b.PNG)

충분히 많은 변수를 제거해주었음에도 불구하고, 여전히 다중공선성이 나타나는 것을 볼 수 있습니다. 따라서 변수 선택법이 아닌 다른 방법을 모색해야합니다.

## Feature Extraction
차원 축소 기법인 PCA를 사용해줍니다. PCA 알고리즘은 기존의 변수들을 linear combination하여 새로운 변수를 만들어 내는 기법입니다. <br><br>

먼저 `MinMaxScaler()`로 정규화를 시켜주겠습니다.
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
arc_scale = pd.DataFrame(scaler.fit_transform(arc_corr), columns=arc_corr.columns)
```

그런 다음 PCA 알고리즘을 적용시키겠습니다. `n_components = 2`는 주성분 개수가 2라는 것을 의미합니다.
```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) # 주성분 개수: 2
pca.fit(arc_scale)
arc_pca = pca.transform(arc_scale)
print(arc_pca.shape)
```

`explained_variance_ratio_`는 pca의 각 주성분이 축소 전 데이터의 설명력을 의미합니다. 즉 1개의 각 주성분이 함축하고 있는 데이터를 의미합니다. 이를 `cumsum()`하여 주성분 개수마다 축소 전 데이터의 어느정도를 설명할 수 있는지(얼마나 덜 손실되었는지)를 알 수 있습니다.
```python
# PCA 주성분분석
pd.Series(np.cumsum(pca.explained_variance_ratio_))
```
```0   0.86267
1   0.92197
dtype: float64
```




## Reference
1. https://bkshin.tistory.com/entry/DATA-20-%EB%8B%A4%EC%A4%91%EA%B3%B5%EC%84%A0%EC%84%B1%EA%B3%BC-VIF
2. https://specialscene.tistory.com/11