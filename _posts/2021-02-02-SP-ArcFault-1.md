---
layout: post
title: '[ArcFault] Learn the basics Part02: Wavelet Transform'
subtitle: 'wavelet transform'
categories: sideproject
tags: arcfault
comments: true
---
`Arc-Fault` 연구과제를 위해 공부한 내용을 정리합니다.

## Introduction
아크 신호를 detection하기 위해 STFT를 사용해 무작위로 생성한 신호를 transform 해주었습니다. 하지만 실제 신호는 더 non-stationary 할 수 있기 때문에 STFT(Short Time Fourier Transform)의 한계점을 보완한 Wavelet Transform을 사용하여 아크 신호를 탐지해보겠습니다.

## PyWavelets
사용한 Python Library는 [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)입니다. import하는 name과 주요 특징 등이 소개되어 있습니다. 이 중에서 제가 사용할 wave transform은 바로 **CWT**(Continous Wavelet Transform)입니다.

## [CWT](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families)
pywt.cwt의 Parameters를 먼저 살펴보겠습니다. <br>
- data: Input signal로 array type입니다.
- scales: window size를 scale array 만큼 각각 적용시켜 각 window size 마다 주파수의 특성을 나타냅니다. 
- wavelet: 사용할 wavelet 기법의 명칭을 뜻합니다.
    - Mexican Hat Wavelet("mexh")
    - Morlet Wavelet("morl")
    - Gaussian Derivative Wavelets("gausP")
    - etc.
<br>

Returns 값
- coefs: 첫 번째 axis는 scales와 일치하는 coefs 계수, 남은 axes에는 data와 shape이 일치한 계수들이 matrix 형식으로 있습니다. frequencies의 값(wavelet 주파수)이 높을 수록 coefs 값도 높은 값을 가집니다.
- frequencies: 인자 값으로 사용한 [continuous wavelet families](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)에 맞게 해당 scale 값 별로 wavelet의 주파수가 출력됩니다. scale값이 커질수록 frequencies의 값은 작아집니다.
- 따라서, transform하고자 하는 신호에 최적화된 scale 값을 찾는 것이 중요합니다.

wavelets의 families를 확인하고 싶을 경우 `wavelist()`를 사용하면 됩니다. <br>
저는 현재 Continous Wavelet Transform을 희망하기 때문에 아래와 같이 kind 인자 값으로 'continuous'를 설정해주면 됩니다.
```python
    pywt.wavelist(kind='continuous')
```

## Python Code

<br>

먼저 PyWavelets, numpy, matplotlib 모듈을 import 해줍니다.<br>
```python
    import pywt
    import numpy as np
    import matplotlib.pyplot as plt

    fs = 100 # sampling rate(sampling frequency)
    t = np.arange(0, 1, 1/fs) # time array
    f1 = 35 # 35Hz
    f2 = 10 # 10Hz

    # amplitude = 0.6, 35Hz의 신호와 amplitude = 2, 10Hz 신호를 합성
    signal = 0.6*np.sin(2*np.pi*f1*t) + 2*np.sin(2*np.pi*f2*t)

    plt.plot(t, signal) # x축: time, y축: Magnitude
    plt.grid()
```
![arcpost04](https://user-images.githubusercontent.com/48666867/106565978-fff73400-6572-11eb-8820-5ba03a413973.PNG)

신호를 다 만들었으면 CWT 기법을 적용시켜 spectrogram을 출력해보겠습니다.

```python
    coef, freqs = pywt.cwt(signal, np.arange(1,50), 'mexh')
    plt.matshow(coef)
    plt.show()
```
![arcpost05](https://user-images.githubusercontent.com/48666867/106566019-0dacb980-6573-11eb-9e1d-2216fa1343ea.PNG)

x축은 time array를 의미하고, y축은 scale 값, 그리고 spectrogram에서 보이는 색상은 window size 즉, scale 값 만큼 shift하여 주파수 탐지하여 나타낸 계수 coefficient 값 입니다.<br>
처음 색이 있는 부분은 전 [포스트](https://geonkimdcu.github.io/sideproject/2021/02/01/SP-ArcFault-0/)에서 알려드린 것 처럼 scale 값이 작을 수록 고주파의 특성을 잘 나타냅니다. 따라서 scale 값이 작은 범위에서 고주파 signal이 출력되었고, scale값이 점점 커져감에 따라 저주파 signal은 약간 희미하거나 혹은 아예 잡히지 않은 것으로 볼 수 있습니다.<br>
결과적으로 현재 입력한 신호에 대해서 최적의 scale값은 5미만인 것으로 볼 수 있습니다.

<br>
이상 Arc-Fault project에서 사용해야할 wavelet transform에 대해 간단히 알아보았습니다. 좀 더 보완하여 Arc-Fault 데이터를 적용시켜 다음 포스트에 업로드 하도록 하겠습니다.

다음 포스트는 Arc-Fault project의 첫 단계에 대해 업로드 하도록 하겠습니다.
감사합니다. :)

<br><br>

## Reference
1. https://github.com/PyWavelets/pywt
2. https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families
