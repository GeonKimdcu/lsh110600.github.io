---
layout: post
title: '[ArcFault] Learn the basics Part01: Signal Transform'
subtitle: 'signal transform'
categories: sideproject
tags: side5
comments: true
---
`Arc-Fault` 연구과제를 위해 공부한 내용을 정리합니다.

## stationary vs non-stationary
**Stationary**(정상성): 시간이 변해도 주기의 변화가 일정한 신호. 즉 통계적 특성이 일정한 시계열입니다. 평균과 분산의 통계적 특성이 동일하면 정상성을 띈다고 봅니다. <br>

**Non-stationary**(비정상성): 시간에 따라 주기의 변화가 불규칙한 신호. 즉 통계적 특성이 변하는 시계열입니다.

![arcpost01](https://user-images.githubusercontent.com/48666867/106413590-ce586d00-648d-11eb-8d64-d254cde6b7d9.PNG) <br>*그림 출처:https://habr.com/en/post/436294/*

<br>

## Fourier Transform
**Fourier transform**은 임의의 입력 신호를 다양한 주파수를 갖는 주기함수들의 합으로 분해하여 표현하는 것입니다. <br>
아래 그림의 예를 들어 설명해보겠습니다. 맨 앞의 붉은색 신호는 입력 신호이고 뒤의 파란색 신호들은 Fourier transform을 통해 얻어진 (원본 신호를 구성하는) 주기함수 성분들입니다. 각각의 주기함수 성분들은 고유의 주파수(frequency)와 강도(amplitude)를 가지고 있으며 **이들을 모두 합치면 원본 붉은색 신호**가 됩니다.
![arcpost02](https://user-images.githubusercontent.com/48666867/106413609-dfa17980-648d-11eb-8f70-f8291ad3a4bc.PNG) <br>*그림 출처: wikipedia*


**Fourier Transform**의 한계란? <br>
일반적으로 모든 시간에 모든 주파수가 혼합되어져 있는 형태로 주파수가 나옵니다. <br>
하지만, 시간마다 주파수가 변한다면 어떻게 될까요?
0~10초까지는 5Hz, 10~20초까지는 10Hz라면 어떻게 될까요?(non-stationary signal)
0~20초까지 5Hz와 10Hz가 공존(합성)하는 신호(stationary signal)와 구별해 낼 수 있을까요?

결론부터 말씀해드리면, Fourier 변환 후 주파수축만이 존재하기에 위와 같은 특성을 이해할 수 없습니다.
따라서 non-stationary signal을 구별해 낼 수 없기 때문에 **Short Time Fourier Transform** (**STFT**)가 나오게 되었습니다.


## STFT
신호를 일정한 크기의 창(window)으로 잘라내서 그 창안의 신호가 stationary signal이 될 때 각 시간 별 주파수를 알아낼 수  있는 방법입니다. <br>
![arcpost03](https://user-images.githubusercontent.com/48666867/106413633-ef20c280-648d-11eb-8609-425c4a7a4c4c.png) <br> *그림 출처: wikipedia*

창의 크기가 작아지면 어떤 주파수가 어떤 시간에 존재하고 있는가를 알기 편해지고, 창의 크기가 커지면 어떤 주파수가 존재하고 있는가를 알기 편해집니다. (창의 크기를 무한대라고 가정하면 창이 없는 Fourier 변환과 같게 됩니다.)<br>
신호에 대해 STFT를 하면 3차원 시간, 주파수, 진폭 그래프가 나오게 됩니다. 하지만 STFT도 문제가 있습니다. 어느 정도의 크기로 잘라야 그 안의 신호가 stationary 해질까요? 이 문제(MultiResolution Problem)를 해결하기 위해 등장한 것이 wavelet 변환입니다.


## [wavelet](https://ko.wikipedia.org/wiki/%EC%9B%A8%EC%9D%B4%EB%B8%94%EB%A6%BF)이란

0을 중심으로 증가와 감소를 반복하는 진폭을 수반한 파동 같은 진동을 말하며, wavelet은 Fourier 변환의 한계를 극복하기 위해 등장했습니다.


## Wavelet Trasform

wavelet transform은 STFT로 stationary 신호를 구별해내기 힘들 때(window로 잘라도 여전히 창 안에 non-stationary 신호가 있을 경우) 사용합니다.<br>

wavelet 변환은 STFT에서의 창에 특정한 wavelet을 넣어 그 창을 스케일(압축하거나 팽창시킴)하여 MultiResolution 문제를 해결할 수 있습니다. <br>
wavelet 변환을 하게 되면 scale, translation, amplitude 축을 가진 3차원 그래프가 나오게 됩니다.<br>
scale은 주파수의 특성을, translation은 translation의 특성을 가집니다. scale은 축척의 의미를 가지고 있으며, wavelet 모함수를 얼마만큼 팽창(압축) 시켰는가를 나타냅니다. scale 값이 커질수록 더욱 많이 팽창시켰다는 의미이므로 저주파의 특성을 나타내고 작을수록 고주파의 특성을 나타냅니다. translation은 모함수 윈도우(wavelet이 포함된 창)의 위치를 나타낸 것입니다. 시간에 따라 창이 이동하는 것(shift)을 이야기합니다.


<br>
이상 Arc-Fault project의 첫 포스트를 마치도록 하겠습니다.<br>
다음 포스트는 이러한 이론을 코드로 구현해보도록 하겠습니다.

<br>
<br>

## Reference
1. https://ktcf.tistory.com/87
2. https://darkpgmr.tistory.com/171
3. https://clavez.tistory.com/54
