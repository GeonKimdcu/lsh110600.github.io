---
layout: post
title: '[DeepLearning] CH05. 컴퓨터 비전을 위한 딥러닝'
subtitle: 'hands-on DeepLearning'
categories: deeplearning
tags: deeplearning
comments: true
---
`케라스 창시자에게 배우는 딥러닝`을 기반으로 공부한 내용을 정리합니다.

<img src="/assets/img/dlcourse/book.jpeg" width="200" height="200">

## 5.1 합성곱 신경망 소개
앞선 장에서 완전 연결 네트워크(densely connected network)로 풀었던 MNIST 숫자 이미지 분류에 컨브넷을 사용해 보겠습니다(이 모델의 test accuracy는 97.8%).

```python
# code 5-1 Create simply ConvNet
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation ='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
```

컨브넷이 (image_height, image_width, image_channels) 크기의 입력 텐서를 사용한다는 점이 중요합니다(배치 차원은 포함하지 않습니다). <br>
MNIST 이미지 포맷인 (28,28,1) 크기의 입력을 처리하도록 컨브넷을 설정해야 합니다. 이 때문에 첫 번째 층의 매개변수로 input_shape=(28, 28, 1)을 전달했습니다.

지금까지의 컨브넷 구조를 출력해보겠습니다.

`model.summary()`
![img](/assets/img/dlcourse/summary01.png)

Conv2D와 MaxPooling2D 층의 출력은 (height, width, channels) 크기의 3D 텐서입니다. 높이와 너비 차원은 네트워크가 깊어질수록 작아지는 경향이 있습니다.<br>
채널의 수는 Conv2D 층에 전달된 첫 번째 매개변수에 의해 조절됩니다(32개 or 64개).

다음 단계에서 마지막 층의 (3, 3, 64) 크기의 출력 텐서를 완전 연결 네트워크에 주입합니다(Dense 층을 쌓은 분류기). 이 분류기는 1D 벡터를 처리하는데, 이전 층의 출력이 3D 텐서입니다. 그래서 3D 출력을 1D 텐서로 펼쳐야 합니다. 그다음 몇 개의 Dense 층을 추가합니다.

```python
# code 5-2 Add Dense layer on ConvNet
model.add(layers.Flatten()) # 1D로 펼침
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

10개의 클래스를 분류하기 위해 마지막 층의 출력 크기를 10으로 하고 소프트 맥스 활성화 함수를 사용합니다.

이제 전체 네트워크 구조를 살펴보겠습니다.

`model.summary()`

![img](/assets/img/dlcourse/summary02.png)
![img](/assets/img/dlcourse/summary03.png)

여기에서 볼 수 있듯이 (3,3,64) 출력이 (576,) 크기의 벡터로 펼쳐진 후 Dense 층으로 주입되었습니다.

이제 MNIST 숫자 이미지에 이 컨브넷을 훈련합니다.

```python
# code 5-3 Train ConvNet to MNIST
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
```

테스트 데이터에서 모델을 평가해보겠습니다.

 `>>> test_loss, test_acc = model.evaluate(test_images, test_labels)` <br>
 `>>>  test_acc` <br>
0.9921

완전 연결 네트워크는 97.8% 테스트 정확도를 얻은 반면에 기본적인 컨브넷은 99.2%의 테스트 정확도를 얻었습니다.

### 5.1.1 합성곱 연산
완전 연결 층과 합성곱 층 사이의 근본적인 차이는 다음과 같습니다. <br>
Dense 층은 입력 특성 공간에 있는 전역 패턴(MNIST 숫자 image에서 모든 픽셀에 걸친 패턴)을 학습하지만 합성곱 층은 지역 패턴을 학습합니다. <br>
이미지일 경우 작은 2D window로 입력에서 패턴을 찾습니다. 앞의 예에선 3 x 3 크기의 window 였습니다.

![img](/assets/img/dlcourse/capture5-1.png)

이 핵심 특징은 컨브넷에 두 가지 흥미로운 성질을 제공합니다.

- 학습된 패턴은 평행 이동 불변성(translation invariant)을 가집니다. <br>

    컨브넷이 이미지의 오른쪽 아래 모서리에서 어떤 패턴을 학습했다면 다른 곳(왼쪽 위 모서리 등)에서도 이 패턴을 인식할 수 있습니다. <br>
    
    완전 연결 네트워크는 새로운 위치에 나타난 것은 새로운 패턴으로 학습해야 합니다. 이런 성질은 컨브넷이 이미지를 효율적으로 처리하게 만들어 줍니다.

    적은 수의 훈련 샘플을 사용해서 일반화 능력을 가진 표현을 학습할 수 있습니다.
![img](/assets/img/dlcourse/capture02.png)

- 컨브넷은 패턴의 공간적 계층 구조를 학습할 수 있습니다. <br>

    첫 번째 합성곱 층이 edge 같은 작은 지역 패턴을 학습합니다. <br>
    두 번째 합성곱 층은 첫 번째 층의 특성으로 구성된 더 큰 패턴을 학습합니다. <br>

    이런 방식을 사용해 컨브넷은 매우 복잡하고 추상적인 시각적 개념을 효과적으로 학습할 수 있습니다.
![img](/assets/img/dlcourse/capture03.png)

합성곱 연산은 **특성 맵**(feature map)이라고 부르는 3D 텐서에 적용됩니다. 이 텐서는 2개의 공간축(높이와 너비)과 깊이 축(채널 축이라고도 합니다)으로 구성됩니다. <br>

RGB 이미지는 3개의 컬러채널(red, greed, blue)을 가지므로 깊이 축의 차원이 3이 됩니다. 흑백 이미지는 깊이 축의 차원이 1입니다.

합성곱 연산은 입력 특성 맵에서 작은 패치(patch)들을 추출하고 이런 모든 패치에 같은 변환을 적용하여 **출력 특성 맵**(output feature map)을 만듭니다.

![img](/assets/img/dlcourse/capture04.png)

우리가 보는 세상은 시각적 구성 요소들의 공간적인 계층 구조로 구성되어 있으며, 아주 좁은 지역의 edge들이 연결되어 눈이나 귀 같은 국부적인 구성 요소를 만들고 이들이 모여서 "고양이"처럼 고수준의 개념을 만듭니다.

![img](/assets/img/dlcourse/capture5-2.png)

output feature map도 높이와 너비를 가진 3D 텐서입니다. 출력 텐서의 깊이는 층의 매개변수로 결정되기 때문에 상황에 따라 다릅니다. <br>
깊이 축의 채널은 더 이상 RGB 입력처럼 특정 컬러를 의미하지 않습니다. <br>
대신 일종의 **filter**를 의미하며 입력 데이터의 어떤 특성을 인코딩합니다. <br>
예를 들어 고수준으로 보면 하나의 필터가 '입력에 얼굴이 있는지'를 인코딩할 수 있습니다.

![img](/assets/img/dlcourse/capture05.png)

MNIST 예제에서는 첫 번째 합성곱 층이 (28, 28, 1) 크기의 특성 맵을 입력으로 받아 (26, 26, 32) 크기의 특성 맵을 출력합니다. <br>
즉 입력에 대해 32개의 필터를 적용합니다. 32개의 출력 채널 각각은 26 x 26 크기의 배열 값을 가집니다. <br>
이 값은 입력에 대한 필터의 **응답 맵**(response map)입니다. 입력의 각 위치에서 필터 패턴에 대한 응답을 나타냅니다. <br>
**특성 맵**은 깊이 축에 있는 다음과 같습니다. 깊이 축에 있는 각 차원은 하나의 특성(또는 필터)이고, 2D 텐서 output[:, :, n]은 입력에 대한 이 필터 응답을 나타내는 2D 공간상의 맵입니다.

다시 말해 feature map은 컨볼루션 필터의 적용 결과로 만들어지는 2차원 행렬입니다. 각 원소는 컨볼루션 필터에 표현된 특징을 대응되는 위치에 포함되고 있는 정도의 값을 나타냅니다.<br>
k개의 컨볼루션 필터를 적용하면 k의 2차원 feature map이 생성됩니다.
![img](/assets/img/dlcourse/capture10.png)

합성곱층은 핵심적인 2개의 파라미터로 정의됩니다. <br>
- 입력으로부터 뽑아낼 패치의 크기: 전형적으로 3 x 3 또는 5 x 5 크기를 사용합니다.
- 특성 맵의 출력 깊이: 합성곱으로 계산할 필터의 수입니다.

케라스의 Conv2D 층에서 이 파라미터는 Conv2D(output_depth, (window_height, window_width))처럼 첫 번째와 두 번째 매개변수로 전달됩니다.

3D 입력 특성 맵 위를 3 x 3 또는 5 x 5 크기의 윈도우가 sliding 하면서 모든 위치에서 3D 특성 패치((window_height, window_width, input_depth) 크기)를 추출하는 방식으로 합성곱이 작동합니다. <br>
이런 3D 패치는 (output_depth, ) 크기의 1D 벡터로 변환됩니다(**합성곱 커널**(convolution kernel)이라고 불리는 하나의 학습된 가중치 행렬과의 텐서 곱셈을 통하여 변환됩니다). 변환된 모든 벡터는 (height, width, output_depth) 크기의 3D 특성 맵으로 재구성됩니다. 

출력 특성 맵의 공간상 위치는 입력 특성 맵의 같은 위치에 대응됩니다(예를 들어 출력의 오른쪽 아래 모서리는 입력의 오른쪽 아래 부근에 해당하는 정보를 담고 있습니다). 3 x 3 윈도우를 사용하면 3D 패치 `input[i-1:i+2, j-1:j+2, :]`로부터 벡터 `output[i, j, :]`가 만들어집니다.

![img](/assets/img/dlcourse/capture06.png)

일정 영역의 값들에 대해 가중치를 적용하여 하나의 값으로 만들어줍니다.

예를 들어 보겠습니다. <br>
아래는 3 x 3 크기의 커널로 5 x 5의 이미지 행렬에 합성곱 연산을 수행하는 과정을 보여줍니다. 한 번의 연산을 1 스텝(step)이라고 하였을 때, 합성곱 연산의 네번째 스텝까지 이미지와 식으로 표현해봤습니다.

![img](/assets/img/dlcourse/capture07.png)

위 연산을 9번의 step까지 마쳤다고 가정할 때, 최종 결과는 아래와 같습니다. 

![img](/assets/img/dlcourse/capture08.png)
위 결과처럼 입력으로부터 커널을 사용해 합성곱 연산을 통해 나온 결과를 **특성 맵**이라고 합니다.

위의 예제에선 커널 크기가 3 x 3 이었지만, 커널의 크기는 사용자가 정할 수 있습니다. 또한 커널의 이동범위가도 정할 수 있습니다. <br>
이러한 이동범 범위를 **스트라이드**(stride)라고 합니다. <br>
스트라이드의 기본 값은 1입니다. 스트라이드 2를 사용했다는 것은 특성 맵의 너비와 높이가 2의 배수로 다운샘플링 되었다는 뜻입니다. <br>
특성 맵을 다운샘플링하기 위해선 스트라이드 대신에 **최대 풀링 연산**(max pooling)을 사용하는 경우가 많습니다.


![img](/assets/img/dlcourse/capture09.png)
아래의 예제는 스트라이드가 2일 경우에 5 × 5 이미지에 합성곱 연산을 수행하는 3 × 3 커널의 움직임을 보여줍니다. 최종적으로 2 × 2의 크기의 특성 맵을 얻습니다.

위의 예에서 5 × 5 이미지에 3 × 3의 커널로 합성곱 연산을 하였을 때, 스트라이드가 1일 경우에는 3 × 3의 특성 맵을 얻었습니다. <br>
이와 같이 합성곱 연산의 결과로 얻은 특성 맵은 입력보다 크기가 작아진다는 특징이 있습니다. 

만약, 합성곱 층을 여러개 쌓았다면 최종적으로 얻은 특성 맵은 초기 입력보다 매우 작아진 상태가 되버립니다. <br>
합성곱 연산 이후에도 특성 맵의 크기가 입력의 크기와 동일하게 유지되도록 하고 싶다면 패딩(padding)을 사용하면 됩니다.
![img](/assets/img/dlcourse/padding.png)

패딩은 (합성곱 연산을 하기 전에) 입력의 가장자리에 지정된 개수의 폭만큼 행과 열을 추가해주는 것을 말합니다. <br>
좀 더 쉽게 설명하면 지정된 개수의 폭만큼 테두리를 추가합니다. 주로 값을 0으로 채우는 제로 패딩(zero padding)을 사용합니다. 

위의 그림은 5 × 5 이미지에 1폭짜리 제로 패딩을 사용하여 위, 아래에 하나의 행을 좌, 우에 하나의 열을 추가한 모습을 보여줍니다.

커널은 주로 3 × 3 또는 5 × 5를 사용한다고 언급한 바 있습니다. <br>
만약 스트라이드가 1이라고 하였을 때, 3 × 3 크기의 커널을 사용한다면 1폭짜리 제로 패딩을 사용하고, 5 × 5 크기의 커널을 사용한다면 2폭 짜리 제로 패딩을 사용하면 입력과 특성 맵의 크기를 보존할 수 있습니다. <br>
예를 들어 5 × 5 크기의 이미지에 1폭짜리 제로 패딩을 하면 7 × 7 이미지가 되는데, 여기에 3 × 3의 커널을 사용하여 1 스트라이드로 합성곱을 한 후의 특성 맵은 기존의 입력 이미지의 크기와 같은 5 × 5가 됩니다.

Conv2D층에서 padding을 매개변수로 설정할 수 있습니다.  <br>
```python
model.add(Conv2D(32, (3,3), padding="valid"))
model.add(Conv2D(32, (3,3), padding="same"))
```
`valid`는 패딩을 사용하지 않겠다는 뜻으로, padding 매개변수의 default 값입니다. <br>
`same`은 입력과 동일한 높이와 너비를 가진 출력을 만들기 위해 패딩을 적용한다는 뜻입니다.

### 5.1.2 3차원 텐서의 합성곱 연산
실제로 합성곱 연산의 입력은 '다수의 채널을 가진' 이미지 또는 이전 연산의 결과로 나온 특성 맵일 수 있습니다. <br>
만약, 다수의 채널을 가진 입력 데이터를 가지고 합성곱 연산을 한다고 하면 커널의 채널 수도 입력의 채널 수만큼 존재해야 합니다. 다시 말해 **입력 데이터의 채널 수와 커널의 채널 수는 같아야 합니다.** 채널 수가 같으므로 합성곱 연산을 채널마다 수행합니다. 그리고 그 결과를 모두 더하여 최종 특성 맵을 얻습니다.

![img](/assets/img/dlcourse/capture11.png)

위 그림은 3개의 채널을 가진 입력 데이터와 3개의 채널을 가진 커널의 합성곱 연산을 보여줍니다. 커널의 각 채널끼리의 크기는 같아야 합니다. <br>
각 채널 간 합성곱 연산을 마치고, 그 결과를 모두 더해서 하나의 채널을 가지는 특성 맵을 만듭니다. <br>
주의할 점은 위의 연산에서 사용되는 커널은 3개의 커널이 아니라 **3개의 채널을 가진 1개의 커널**이라는 점입니다.

위 그림은 높이 3, 너비 3, 채널 3의 입력이 높이 2, 너비 2, 채널 3의 커널과 합성곱 연산을 하여 높이 2, 너비 2, 채널 1의 특성 맵을 얻는다는 의미입니다. 합성곱 연산의 결과로 얻은 특성 맵의 채널 차원은 RGB 채널 등과 같은 컬러의 의미를 담고 있지는 않습니다.

### 5.1.3 최대 풀링 연산
스트라이드 합성곱과 매우 비슷하게 강제적으로 feature map을 다운샘플링하는 것이 최대 풀링의 역할입니다. <br>
예시로 컨브넷 예제에서 첫 번째 MaxPooling2D 층 이전에 특성 맵의 크기는 26 x 26 이었는데 최대 풀링 연산으로 13 x 13으로 줄어들었습니다.

최대 풀링은 입력 특성 맵에서 윈도우에 맞는 패치를 추출하고 각 채널별로 최댓값을 출력합니다. <br>
합성곱과 개념적으로 비슷하지만 추출한 패치에 학습된 선형 변환(합성곱 커널)을 적용하는 대신 하드코딩된 최댓값 추출 연산을 사용합니다.

합성곱과 가장 큰 차이점은 최대 풀링은 보통 2 x 2 window와 스트라이드 2를 사용하여 특성 맵을 절반 크기로 다운샘플링한다는 것입니다. <br>
이에 반해 합성곱은 전형적으로 3 x 3 window와 스트라이드 1을 사용합니다.

보통 합성곱 층(합성곱 연산 + 활성화 함수) 다음에는 풀링 층을 추가하는 것이 일반적입니다. <br>
풀링 층에서는 특성 맵을 다운샘플링하여 특성 맵의 크기를 줄이는 풀링 연산이 이루어집니다. 

풀링 연산에는 일반적으로 최대 풀링(max pooling)과 평균 풀링(average pooling)이 사용됩니다. 우선 최대 풀링을 통해서 풀링 연산을 이해해봅시다.

![img](/assets/img/dlcourse/pooling.png)

풀링 연산에서도 합성곱 연산과 마찬가지로 커널과 스트라이드의 개념을 가집니다. <br>
위의 그림은 스트라이드가 2일 때, 2 × 2 크기 커널로 맥스 풀링 연산을 했을 때 특성맵이 절반의 크기로 다운샘플링되는 것을 보여줍니다.<br>
맥스 풀링은 커널과 겹치는 영역 안에서 최대값을 추출하는 방식으로 다운샘플링합니다.

다른 풀링 기법인 평균 풀링은 최대값을 추출하는 것이 아니라 평균값을 추출하는 연산이 됩니다. <br>

풀링 연산은 커널과 스트라이드 개념이 존재한다는 점에서 합성곱 연산과 유사하지만, 합성곱 연산과의 차이점은 학습해야 할 가중치가 없으며 연산 후에 채널 수가 변하지 않는다는 점입니다.

풀링을 사용하면, 특성 맵의 크기가 줄어드므로 특성 맵의 가중치의 개수를 줄여줍니다.

<br><br>

## Reference
1. 케라스 창시자에게 배우는 딥러닝
2. https://wikidocs.net/64066