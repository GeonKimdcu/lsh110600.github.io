---
layout: post
title: '[DeepLearning] CH02. 신경망의 수학적 구성 요소'
subtitle: 'deepLearning start'
categories: deeplearning
tags: deeplearning
comments: true
---
`케라스 창시자에게 배우는 딥러닝`을 기반으로 공부한 내용을 정리합니다.

<img src="/assets/img/dlcourse/book.jpeg" width="200" height="200">

## 2.1 신경망과의 첫 만남
먼저 딥러닝계의 "hello wrold"라고 할 수 있는 MNIST를 사용하여 신경망에 대해 알아보겠습니다.

[MNIST 데이터베이스](https://ko.wikipedia.org/wiki/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4) (Modified National Institute of Standards and Technology database)는 손으로 쓴 숫자들로 이루어진 대형 데이터베이스이며, 다양한 화상 처리 시스템을 트레이닝하기 위해 일반적으로 사용됩니다.  <br>
흔히 흑백 손글씨 숫자 이미지(28 * 28 pixels)를 10개의 범주(0에서 9까지)로 분류하는데 이용할 수 있으며, 6만개의 트레이닝 이미지와 1만개의 테스트 이미지로 구성되어 있습니다.
![mnist](/assets/img/dlcourse/MnistExamples.png)

> 머신러닝에서 분류 문제의 범주(category)를 클래스(class)라고 합니다. 데이터 포인트는 샘플(sample)이라고 합니다. 특정 샘플의 클래스는 레이블(label)이라고 합니다.

MNIST 데이터 셋은 numpy array 형태로 케라스에 이미 포함되어 있습니다.

```python
# code 2-1 케라스에서 MNIST DataSet 적재하기
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

train_images와 train_labels가 모델이 학습해야 할 training set을 구성합니다. 모델은 test_images와 test_labels로 구성된 test set에서 테스트 될 것입니다. image는 넘파이 배열로 인코딩되어 있고 label은 0부터 9까지의 숫자 배열입니다. 이미지와 배열은 일대일 관계입니다.

먼저 훈련 데이터를 살펴보겠습니다.
> `train_images.shape` <br>
(60000, 28, 28)

> `len(train_labels)` <br>
(60000)

> `train_labels` <br>
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

다음은 테스트 데이터입니다.

> `test_images.shape` <br>
(10000, 28, 28)

> `len(test_labels)` <br>
10000

작업 순서는 다음과 같습니다.
- train_images와 train_labels를 네트워크에 주입합니다.
- 네트워크는 이미지와 레이블을 연관시킬 수 있도록 학습됩니다.
- test_images에 대한 에측을 네트워크에 요청합니다.
- 예측이 test_labels와 맞는지 확인합니다.

이제 신경망을 만들어 보겠습니다.

```python
# code 2-2 신경망 구조
from keras import models
from kerars import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
```
이 신경망의 구조를 그림으로 나타내면 다음과 같습니다.
![신경망](/assets/img/dlcourse/IMG_0124.jpg)


신경망의 핵심 구성 요소는 일종의 데이터 처리 필터와 같은 층(layers)입니다. 어떤 데이터가 들어가면 더 유용한 형태로 출력됩니다. <br>
구체적으로 층은 주어진 문제에 더 의미있는 표현(representation)을 입력된 데이터로부터 추출합니다. <br>
대부분의 딥러닝은 간단한 층을 연결하여 구성되어 있고, 점진적으로 데이터를 정제하는 형태를 띠고 있습니다. 딥러닝 모델은 데이터 정제 필터(layer)가 연속되어 있는 데이터 프로세싱을 위한 여과기와 같습니다.

이 예에서는 조밀하게 연결된(또는 **완전 연결**(fully connected)된) 신경망 층인 **Dense** 층 2개가 연속되어 있습니다. 두 번째 (즉 마지막) 층은 10개의 확률 점수가 들어 있는 배열(모두 더하면 1)을 반환하는 **소프트맥스**(softmax)층입니다. 각 점수는 현재 숫자 이미지가 10개의 숫자 클래스 중 하나에 속할 확률입니다.

신경망이 훈련 준비를 마치기 위해서 컴파일 단계에 포함될 세 가지가 더 필요합니다.
- **손실 함수**(loss function): 훈련 데이터에서 신경망의 성능을 측정하는 방법으로 네트워크가 옳은 방향으로 학습될 수 있도록 도와줍니다.

- **옵티마이저**(optimizer): 입력된 데이터와 손실 함수를 기반으로 네트워크를 업데이트하는 메커니즘입니다.

- **훈련과 테스트 과정을 모니터링할 지표**: 정확도(정확히 분류된 이미지의 비율)고려

```python
# code 2-3 컴파일 단계
network.compile(optimizer='rmsprop', 
                loss ='categorical_crossentropy',
                metrics=['accuracy'])
```

**훈련을 시작하기 전에 데이터를 네트워크에 맞는 크기로 바꾸고 모든 값을 0과 1 사이로 스케일을 조정해야 합니다.**

훈련 이미지 데이터를 0과 1사이의 값을 가지는 float32 타입의 (60000, 28 * 28) 크기인 배열로 바꿔줍니다.

```python
# code 2-4 이미지 데이터 준비하기
train_images= train_images.reshape((60000, 28 * 28)) # 3D -> 2D
train_images = train_images.astype('float32') / 255 # 정규화

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```
또 레이블을 범주형으로 인코딩해줘야 합니다.

```python
# code 2-5 레이블 준비하기
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```
이제 신경망을 훈련시킬 준비가 되었습니다. 케라스에서는 fit 메서드를 호출하여 훈련 데이터에 모델을 학습시킵니다.

> network.fit(train_images, train_labels, epochs = 5, batch_size = 128) <br>
Epoch 1/5 <br>
60000/60000 [====================] - 1s 22us/step - loss: 0.2571 - acc:0.9257
Epoch 2/5 <br>
60000/60000 [====================] - 1s 12us/step - loss: 0.1027 - acc:0.9696
Epoch 3/5 <br>
60000/60000 [====================] - 1s 12us/step - loss: 0.0686 - acc:0.9696
Epoch 4/5 <br>
60000/60000 [====================] - 1s 12us/step - loss: 0.0494 - acc:0.9856
Epoch 5/5 <br>
60000/60000 [====================] - 1s 12us/step - loss: 0.0368 - acc:0.92895

훈련 데이터에 대한 네트워크의 손실과 정확도가 출력됩니다.

이제 테스트 세트에서 모델이 잘 작동하는지 확인해 보겠습니다.

> `test_loss, test_acc = network.evaluate(test_images, test_labels) `<br>
10000/10000 [====================] - 0s 16us/step

> `print('test_acc:', test_acc)` <br>
test_acc: 0.9789

테스트의 정확도는 97.8%로 나왔습니다. 훈련 세트 정확도보다는 약간 낮은데 이는 [과대적합](https://ko.wikipedia.org/wiki/%EA%B3%BC%EC%A0%81%ED%95%A9)(overfitting)때문입니다. 이는 머신 러닝 모델이 훈련 데이터보다 새로운 데이터에서 성능이 낮아지는 경향을 뜻합니다.



<br><br>
이번 장에서는 신경망을 생성하고 훈련시키는 전체적인 흐름을 파악하는 시간을 가졌습니다.

다음 장에서는 텐서, 신경망에 주입하는 데이터의 저장 형태, 층을 만들어 주는 텐서 연산, 신경망을 훈련 샘플로부터 학습시키는 경사 하강법에 대해 알아보겠습니다.

<br><br>

## Reference
1. 케라스 창시자에게 배우는 딥러닝
