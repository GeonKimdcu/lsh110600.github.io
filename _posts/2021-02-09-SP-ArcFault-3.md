---
layout: post
title: '[ArcFault] Project step02: Data Analysis'
subtitle: 'data analysis'
categories: sideproject
tags: arcfault
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

`explained_variance_ratio_`는 pca의 각 주성분이 축소 전 데이터의 설명력을 의미합니다. 즉 1개의 각 주성분이 함축하고 있는 데이터를 의미합니다. 이를 `cumsum()`하여 주성분 개수마다 축소 전 데이터의 어느 정도를 설명할 수 있는지(얼마나 덜 손실되었는지)를 알 수 있습니다.
```python
# PCA 주성분분석
pd.Series(np.cumsum(pca.explained_variance_ratio_))
```
**출력 결과** <br>
> 0　　0.86267 <br>
> 1　　0.92197

출력 결과를 보아 첫 번째 주성분은 원 데이터의 86%를 설명할 수 있으며, 두 개의 주성분을 가졌을 땐 92%의 설명력을 가진다고 할 수 있습니다.

그 후 두 개의 주성분을 사용하여 차원 축소된 데이터와 라벨을 합쳐 데이터프레임을 생성해줍니다.

## Imbalanced Classes
실제 아크 상태와 정상 상태의 클래스 분포 비율은 정상 상태가 압도적으로 많습니다. 즉 클래스 분포가 불균형적입니다. 

예를 들어 설명해보겠습니다. 암 환자를 분석한다고 했을 경우 암에 걸린 환자와 암에 걸리지 않은 환자의 클래스 분포 비율은 당연 암에 걸리지 않은 환자의 비율이 훨씬 많을 것 입니다. 따라서 이러한 클래스의 불균형을 해소해주어야 합니다.

먼저 target variable 분포를 확인해보겠습니다.
```python
from collections import Counter

Counter(arc_pca.label)
```
**출력 결과** <br>
> Counter({0: 20, 1: 40})

현재 적은 양의 데이터로 진행하기 때문에 이 정도의 불균형은 크게 문제가 되지 않으나, 실제 데이터 수집을 하고 나서는 불균형하기 때문에 이를 SMOTE 기법을 이용하여 Oversampling 해주겠습니다.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# 기존의 X_train, y_train, X_test, y_test의 형태확인
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
```

형태를 확인했으니 이제 Oversampling 해주겠습니다.

```python
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {}".format(sum(y_train == 0)))

sm = SMOTE(random_state = 42, sampling_strategy = 'auto') # SMOTE 알고리즘, 비율 증가
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) # Over Sampling 진행

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
```

Oversampling 적용 후 label '1'과 '0'의 개수가 같아진 것을 확인 할 수 있습니다.

train set의 형태가 어떻게 바뀌었는지 확인해보겠습니다.

```python
print("Before OverSampling, the shape of X_train: {}".format(X_train.shape)) # SMOTE 적용 이전 데이터 형태
print("Before OverSampling, the shape of y_train: {}".format(y_train.shape)) # SMOTE 적용 이전 데이터 형태
print("After OverSampling, the shape of X_train: {}".format(X_train_res.shape)) # SMOTE 적용 결과 확인
print("After OverSampling, the shape of y_train: {}".format(y_train_res.shape)) # SMOTE 적용 결과 확인
```

## SVM(Support Vector Machine)
[SVM(Support Vector Machine)](https://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0)이란 주어진 데이터가 어느 카테고리에 속할지 판단하는 이진 선형 분류 모델입니다.

먼저 features와 target 변수를 설정해줍니다.
```python
features = X_train_res
target = y_train_res
```

다음으로 SVM 모델을 생성 후 학습을 시켜보겠습니다.
```python
from sklearn.svm import SVC
from sklearn import svm, metrics
import numpy as np
import matplotlib.pyplot as plt

svc = SVC(kernel = 'linear', class_weight = 'balanced', C = 1.0, random_state = 0)
model = svc.fit(features, target) # SVM 모델 학습
```

다음은 혼동 행렬을 출력해보겠습니다.
```python
from sklearn.metrics import confusion_matrix
y_pred = svc.predict(features)
confusion_matrix(target, y_pred)
```

## kernel SVM 적합 및 비교
다음으로 kernel 별 적합한 SVM 모델을 찾아보겠습니다. <BR>
- `LinearSVC`, `radial basis function`, `polynomial kernel` 이 있습니다.

우선 시각화 함수를 생성해보겠습니다.
```python
def make_meshgrid(x, y, h = .02):
    x_min, x_max = x.min()-1, x.max() + 1
    y_min, y_max = y.min()-1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
```

다음으로 모델을 정의하고 학습을 시켜줍니다.
```python
C = 1.0 # Regularization parameter
models = (svm.SVC(kernel = 'linear', C=C),
         svm.LinearSVC(C=C, max_iter = 10000),
         svm.SVC(kernel = 'rbf', gamma = 0.7, C=C),
         svm.SVC(kernel = 'poly', degree = 3, gamma = 'auto', C=C))
models = (clf.fit(X, y) for clf in models)
```

이제 kernel 별 어떻게 분류가 되는지 시각화하여 나타내보겠습니다.
```python
titles = ('SVC with linear kernel',
         'LinearSVC (linear kernel)',
         'SVC with RBF kernel',
         'SVC with polynomial (degree 3) kernel')

fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap = plt.cm.coolwarm, alpha = 0.8)
    ax.scatter(X0, X1, c = y, cmap=plt.cm.coolwarm, s = 20, edgecolors = 'k')
    ax.set_xlim(xx.min(), xx.max())
    #ax_set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_title(title)

plt.show()
```

![arcpost17](https://user-images.githubusercontent.com/48666867/108459616-5629de00-72ba-11eb-8748-837fc27df4ec.PNG)

시각화 결과를 보면 데이터 개수가 적어 대부분 잘 분류되는 것을 확인할 수 있습니다. 그리고 아크 상태(Arc1, Arc2)와 normal 상태가 잘 분류되는 것을 볼 수 있습니다.
<br><br>

## GridSearch

다음으로 Hyperparameter optimizationd의 방법 중 하나인 [GridSearch](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)를 통해 최적의 파라미터 값을 탐색해보겠습니다. 

먼저 그리스 서치 매개변수를 설정해준 후, 그리드 서치를 수행하고 테스트 데이터의 accuracy를 확인해보겠습니다.

```python
from sklearn import svm, metrics, model_selection
from sklearn.model_selection import GridSearchCV

print("학습 데이터의 수 =", len(target))

# 그리드 서치 매개변수 설정
params = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},
    {"C" : [1, 10, 100, 1000], "kernel" : ["rbf"], "gamma":[0.001, 0.0001]}
]

# 그리드 서치 수행
clf = GridSearchCV(svm.SVC(), params, n_jobs = -1)
clf.fit(features, target)
print("학습기 =", clf.best_estimator_)

# 테스트 데이터 확인하기
pre = clf.predict(X_test)
ac_score = metrics.accuracy_score(pre, y_test)
print("정답률 =", ac_score)
```
출력 결과로 학습 데이터의 수, 모델의 accuracy, 학습기에는 가장 최적의 파라미터 값이 나옵니다.

모델 성능도 한 번 살펴보겠습니다.
```python
svc = SVC(kernel = 'linear', class_weight = 'balanced', C = 1.0, random_state = 0)
model = svc.fit(features, target)

pre = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, pre)
cl_report = metrics.classification_report(y_test, pre)
print("정답률 = ",ac_score)
print("리포트 =\n", cl_report)
```

Class를 예측하고자 하는 경우엔 shape을 맞춰 값을 할당한 후 predict해주면 됩니다.

```python
new_observation = [[1, -0.2]]
svc.predict(new_observation)
```
<br>
이상으로 Arc-Fault 데이터 분석 후 분류 모델 생성까지 마쳤습니다.

다음으로 시계열 데이터를 통해 arc 상태를 탐지해보는 시간을 가져보겠습니다.

감사합니다 :)

<br><br>

## Reference
1. https://bkshin.tistory.com/entry/DATA-20-%EB%8B%A4%EC%A4%91%EA%B3%B5%EC%84%A0%EC%84%B1%EA%B3%BC-VIF
2. https://specialscene.tistory.com/11