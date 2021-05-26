---
layout: post
title: '[DeepLearning] CH04. 머신 러닝의 기본 요소(4)'
subtitle: 'deepLearning start'
categories: deeplearning
tags: deeplearning
comments: true
---
`케라스 창시자에게 배우는 딥러닝`을 기반으로 공부한 내용을 정리합니다.

<img src="/assets/img/dlcourse/book.jpeg" width="200" height="200">

## 4.4 과대적합과 과소적합
**최적화**(optimization)는 가능한 훈련 데이터에서 최고의 성능을 얻으려고 모델을 조정하는 과정입니다. <br>
반면에 **일반화**(generalization)는 훈련된 모델이 이전에 본 적 없는 데이터에서 얼마나 잘 수행되는지 의미합니다.<br>
모델을 만드는 목적은 좋은 일반화 성능을 얻는 것입니다. 하지만 일반화 성능을 제어할 방법이 없습니다. 단지 훈련 데이터를 기반으로 모델을 조정할 수만 있습니다.

훈련 초기에 최적화와 일반화는 상호 연관되어 있습니다. 훈련 데이터의 손실이 낮아질수록 테스트 데이터의 손실도 낮아집니다. 이때 모델이 **과소적합**(underfitting)되었다고 말합니다. <br>
모델의 성능이 계속 발전될 여지가 있습니다. 즉, 네트워크가 훈련 데이터에 있는 관련 특성을 모두 학습하지 못했습니다. <Br>
하지만 훈련 데이터에 여러 번 반복 학습하고 나면 어느 시점부터 일반화 성능이 더 이상 높아지지 않습니다. 검증 세트의 성능이 멈추고 감소되기 시작합니다. 즉 모델이 과대적합되기 시작합니다.<br>
이는 훈련 데이터에 특화된 패턴을 학습하기 시작했다는 의미입니다. 이 패턴은 새로운 데이터와 관련성이 적어 잘못된 판단을 하게 만듭니다. <br>

모델이 관련성이 없고 좋지 못한 패턴을 훈련 데이터에서 학습하지 못하도록 하려면 가장 좋은 방법은 `더 많은 훈련 데이터를 모으는 것` 입니다. 더 많은 데이터에서 훈련된 모델은 자연히 일반화 성능이 더욱 뛰어납니다. 

만약 데이터를 더 모으는 것이 불가능할 땐 모델이 수용할 수 있는 정보의 양을 조절하거나 저장할 수 있는 정보에 제약을 가하는 것입니다. 네트워크가 적은 수의 패턴만 기억할 수 있다면 최적화 과정에서 가장 중요한 패턴에 집중하게 될 것입니다. 이런 패턴은 더 나은 일반화 성능을 제공할 수 있습니다.

이런 식으로 과대적합을 피하는 처리 과정을 **규제**(regularization)라고 합니다.

### 4.4.1 네트워크 크기 축소
과대적합을 막는 가장 단순한 방법은 모델의 크기, 즉 모델에 있는 학습 파라미터의 수를 줄이는 것입니다.<br>
파라미터의 수는 층의 수와 각 층의 유닛 수에 의해 결정됩니다. <br>
딥러닝에서 모델에 있는 학습 파라미터의 수를 모델의 용량(capacity)이라고 말합니다. 당연히 파라미터가 많은 모델이 기억 용량이 더 많습니다. <br>
이런 매핑은 일반화 능력이 없습니다. <br>
항상 유념해야 할 것은 딥러닝 모델은 **훈련 데이터**에 잘 맞추려는 경향이 있다는 점입니다. <Br>
하지만 진짜 문제는 최적화가 아니고 일반화입니다.

다른 한편으로 네트워크가 기억 용량에 제한이 있다면 이런 매핑을 쉽게 학습하지 못할 것입니다.<br>
따라서 손실을 최소화하기 위해 타깃에 대한 예측 성능을 가진 압축된 표현을 학습해야 합니다. 동시에 과소적합되지 않도록 충분한 파라미터를 가진 모델을 사용해야 합니다ㅏ. <br>
모델의 기억 용량이 부족해서는 안되며, 너무 많은 용량과 충분하지 않은 용량 사이의 절충점을 찾아야 합니다.

데이터에 알맞은 모델 크기를 찾으려면 각기 다른 구조를(검증세트에서) 평가해 보아야 합니다. 적절한 모델 크기를 찾은 일반적인 작업 흐름은 비교적 적은 수의 층과 파라미터로 시작합니다. 그다음 검증 손실이 감소되기 시작할 때까지 층이나 유닛의 수를 늘리는 것입니다.

예시로 영화 리뷰 분류 모델을 살펴보겠습니다.

```python
# code 4-3 original model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', imput_shape=(1000.,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

```python
# code 4-4 smaller model
model = models.Sequential()
model.add(layers.Dense(6, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

점으로 표현된 것이 작은 네트워크, 덧셈 기호가 원래 네트워크입니다(Validation loss가 작은 것이 좋은 모델입니다).

![img](/assets/img/dlcourse/small.png)

작은 네트워크가 기본 네트워크보다 더 나중에 과대적합되기 시작했습니다. 과대적합이 시작되었을 때 성능이 더 천천히 감소되었습니다.

```python
# code 4-5 bigger model
model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

점은 용량이 큰 네트워크의 검증 손실이고, 덧셈 기호는 원본 네트워크의 검증 손실입니다.

![img](/assets/img/dlcourse/bigger.png)

용량이 큰 네트워크는 첫 번째 에포크 이후 거의 바로 과대적합이 시작되어 갈수록 더 심해집니다. 검증 손실도 매우 불안정합니다.

![img](/assets/img/dlcourse/big.png)

한편 두 네트워크의 훈련 손실을 보겠습니다. 용량이 큰 네트워크는 훈련 손실이 매우 빠르게 0에 가까워집니다. <br>
즉 용량이 많은 네트워크일수록 더 빠르게 훈련 데이터를 모델링할 수 있습니다(결국 trianing loss가 낮아집니다). <br>
하지만 더욱 과대적합에 민감해집니다.

### 4.4.2 가중치 규제 추가
**오캄의 면도날**(Occam`s razor)은 어떤 것에 대한 두 가지의 설명이 있을 경우, 더 적은 가정이 필요한 간단한 설명이 옳을 것이라는 이론입니다. <br>
이를 신경망에 적용시켜, 어떤 훈련 데이터와 네트워크 구조가 주어졌을 때 데이터를 설명할 수 있는 가중치 값의 집합은 여러 개입니다. 간단한 모델이 복잡한 모델보다 덜 과대적합될 가능성이 높습니다.

여기에서 간단한 모델은 파라미터 값 분포의 엔트로피가 작은 모델입니다(또는 적은 수의 파라미터를 가진 모델). <br>
그러므로 과대적합을 완화하기 위한 일반적인 방법은 네트워크의 복잡도에 제한을 두어 가중치가 작은 값을 가지도록 강제하는 것입니다. 가중치 값의 분포가 더 균일하게 됩니다.<br> 이를 **가중치 규제**(weight regularization)라고 하며, 네트워크의 손실 함수에 큰 가중치에 연관된 비용을 추가합니다.
- **L1 규제**: 가중치의 절댓값에 비례하는 비용이 추가됩니다(가중치의 L1 노름(norm)).
- **L2 규제**: 가중치의 제곱에 비례하는 비용이 추가됩니다(가중치의 L2 노름). 다른 말로 **가중치 감쇠**(weight decay)라고도 부릅니다.

예시로 영화 리뷰 분류 네트워크에 L2 가중치 규제를 추가해보겠습니다.

```python
#code 4-6 Add L2 weight from model
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                        activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                        activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```

l2(0.001)는 가중치 행렬의 모든 원소를 제곱하고 0.001을 곱하여 네트워크의 전체 손실에 더해진다는 의미입니다. <br>
이 페널티 항은 훈련할 때만 추가됩니다. 이 네트워크의 손실은 테스트보다 훈련할 때 더 높을 것입니다.

![img](/assets/img/dlcourse/l2norm.png)

두 모델이 동일한 파라미터 수를 가지고 있더라도 L2 규제를 사용한 모델이 기본 모델보다 훨씬 더 과대적합에 잘 견디고 있습니다.

케라스에서는 L2 규제 대신에 다음 가중치 규제 중 하나를 사용할 수 있습니다.
```python
# code 4-7 weight regularization used to Keras
from keras import regularizers

regularizers.l1(0.001) # L1 규제
regularizers.l1_l2(l1=0.001, l2=0.001) # L1과 L2 규제 병행
```

### 4.4.3 드롭아웃 추가
**드롭아웃**(dropout)은 신경망을 위해 사용되는 규제 기법 중에서 가장 효과적이고 널리 사용된느 방법 중 하나입니다. <br>
네트워크 층에 드롭아웃을 적용하면 훈련하는 동안 무작위로 층의 일부 출력 특성을 제외시킵니다(0으로 만듭니다). 

한 층이 정상적으로 훈련하는 동안 어떤 입력 샘플에 대해 `[0.2, 0.5, 1.3, 0.8, 1.1]` 벡터를 출력한다고 가정하겠습니다. <br>
dropout을 적용하면 이 벡터의 일부가 random하게 0으로 바뀝니다. `[0, 0.5, 1.3, 0, 1.1]`. <br>

드롯아웃 비율은 0이 될 특성의 비율입니다. 보통 0.2 ~ 0.5로 지정합니다. <br>
테스트 단계에서는 어떤 유닛도 드롭아웃되지 않습니다. 그 대신에 층의 출력을 드롭아웃 비율에 비례하여 줄여 줍니다. 훈련할 때보다 더 많은 유닛이 활성화되기 때문입니다.

더 깊이 파고들어 보겠습니다.<br>
> 5개의 은닉 유닛(hidden unit)을 갖는 한개의 은닉층을 사용하는 다층 퍼셉트론(multilayer perceptron) 의 예를 다시 들어보겠습니다. <br> 
<br>
이 네트워크의 아키텍처는 다음과 같이 표현됩니다.

![img](/assets/img/dlcourse/drop_network.png)

은닉층에 드롭아웃(dropout)을 확률 p로 적용하는 경우, 은닉 유닛들을  p  확률로 제거하는 것이 됩니다. <br>
이유는, 그 확률을 이용해서 출력을 0으로 설정하기 때문입니다.<br> 여기서  h2  와  h5  가 제거되었습니다. <br>
결과적으로  y  를 계산할 때,  h2  와  h5  는 사용되지 않게 되고, 역전파(backprop)를 수행할 때 이것들에 대한 그래디언트(gradient)들도 적용되지 않습니다.<br>
이렇게 해서 출력층을 계산할 때  h1~h5  중 어느 하나에 전적으로 의존되지 않게 합니다. 

![img](/assets/img/dlcourse/drop_layer.png)

이것이 overfitting을 해결하는 정규화(regularization) 목적을 위해서 필요한 것입니다.<br>
테스트 시에는, 더 확실한 결과를 얻기 위해서 dropout을 사용하지 않는 것이 일반적입니다.

다시 돌아와, 크기가 `(batch_size, features)`인 한 층의 출력을 담고 있는 넘파이 행렬을 생각해 보겠습니다. 훈련할 때는 이 행렬 값의 일부가 랜덤하게 0이 됩니다.

```python
# 훈련할 때 유닛의 출력 중 50%를 버립니다.
layer_output = np.random.randint(0, hight = 2, size=layer_output.shape)
```

테스트할 때는 dropout 비율로 출력을 낮추어 주어야 합니다. 
```python
# test 단계
layer_output *= 0.5
```

훈련 단계에 이 두 연산을 포함시켜 테스트 단계에는 출력을 그대로 두도록 구현할 수 있습니다.

```python
layer_output *= np.random.randint(0, high=2, size=layer_output.shape) # 훈련 단계
layer_output /= 0.5 # 스케일 낮추는 대신 높입니다.
```

![img](/assets/img/dlcourse/dropout.png)

드롭아웃의 핵심 아이디어는 층의 출력 값에 noise를 추가하여 중요하지 않은 우연한 패턴을 깨뜨리는 것입니다. noise가 없다면 네트워크가 이 패턴을 기억하기 시작할 것입니다.

케라스에서는 층의 출력 바로 뒤에 dropout 층을 추가하여 네트워크에 드롭아웃을 적용할 수 있습니다.

```python
# code 4-8 Add Dropout to IMDB network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```

결과 그래프를 보면 기본 네트워크보다 확실히 향상되었습니다.

![img](/assets/img/dlcourse/dropout_model.png)

정리하면 신경망에서 과대적합을 방지하기 위해 가장 널리 사용하는 방법은 다음과 같습니다.

1. 훈련 데이터를 더 모읍니다(가장 쉬우며, 일반화 성능이 증가합니다).
2. 네트워크의 용량을 감소시킵니다.
3. 가중치 규제를 추가합니다.
4. 드롭아웃을 추가합니다.

<br><br>

## Reference
1. 케라스 창시자에게 배우는 딥러닝
2. https://ko.d2l.ai/chapter_deep-learning-basics/dropout.html