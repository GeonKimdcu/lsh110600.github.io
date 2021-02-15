---
layout: post
title: '[Python] Tensorflow windows 10 GPU Install'
subtitle: 'Tensorflow windows 10 GPU Install'
categories: programing
tags: python
comments: true
---
deep learning을 위한 `tensorflow GPU`를 설치합니다.

## Introduction
텐서플로우(TensorFlow) 2.0 를 윈도우(Windows) 10에서 GPU를 활용하여 학습할 수 있도록 설치하는 방법에 대하여 공유드리고자 합니다.

그래픽카드는 반드시 **NVIDIA 계열의 그래픽 카드**가 탑재되어 있어야 합니다.

- 작성일 기준 버전
    - 설치: TensorFlow 2.2.0
    - 그래픽카드: NVidia GTX 1060
    - OS: Windows 10

Ref.[텐서플로우 공식 GPU 설치 가이드](https://www.tensorflow.org/install/gpu)

## Step 1. 텐서플로우(TensorFlow) 설치
pip 명령어를 통해 텐서플로우를 설치합니다.<br>
2.0 이상의 텐서플로우는 별도의 GPU 버전을 따로 설치할 필요가 없습니다.
> TensorFlow 2.0 이상의 버전 설치 <br>
`pip install tensorflow`

## Step 2. NIVIDA GPU 드라이버 설치
[NVIDIA GPU 드라이버 설치 링크](https://www.nvidia.com/download/index.aspx?lang=kr)


- 제품 유형, 시리즈, 계열: 자신의 그래픽 카드 정보를 선택합니다.
- 운영체제: Windows 10을 선택하며, bit는 32/64 중 자신의 os와 일치된 bit 운영체제를 선택합니다.
- 다운로드 타입: Game Ready 드라이버 혹은 Studio 드라이버를 선택합니다. (큰 상관 없습니다)

선택을 완료 하셨다면, 검색을 클릭합니다.

![post1](https://user-images.githubusercontent.com/48666867/107910260-6893d700-6f9d-11eb-8109-8a54c1cf1664.png)

TensorFlow GPU 설치를 위해서는 **418.x 버전 이상**이 요구됩니다.

![post2](https://user-images.githubusercontent.com/48666867/107910432-c32d3300-6f9d-11eb-8e39-846dbf387956.png)

사용자 정의 설치를 해줍니다.
![post3](https://user-images.githubusercontent.com/48666867/107910590-169f8100-6f9e-11eb-9cde-6bc0f9850769.png)

구성 요소를 선택합니다.
![post4](https://user-images.githubusercontent.com/48666867/107910651-2fa83200-6f9e-11eb-8c49-9ab874bb9d88.png)

설치 완료 후, 터미널에 `nvidia-smi` 명령어를 입력해줍니다.
정상적으로 NVIDIA 그래픽 드라이버가 설치 되었는지 확인합니다.

CUDA Version을 확인합니다.
![post5](https://user-images.githubusercontent.com/48666867/107910825-8877ca80-6f9e-11eb-9b03-ff16b9f1cbb6.PNG)


## STEP 3. CUDA Toolkit 다운로드 및 설치
[CUDA Toolkit 다운로드 링크](https://developer.nvidia.com/cuda-toolkit-archive)

위 링크에서 자신이 맞는 버전의  CUDA Toolkit을 다운로드 합니다.<br><br>

![post6](https://user-images.githubusercontent.com/48666867/107911133-308d9380-6f9f-11eb-845e-f8561a3f7a05.PNG)

다운로드 받은 exe를 실행하고 설치를 진행해줍니다.

![post7](https://user-images.githubusercontent.com/48666867/107912050-1a80d280-6fa1-11eb-8dd0-31135a8389ee.PNG)

NVIDIA GeForce Experience 체크 해제
![post8](https://user-images.githubusercontent.com/48666867/107912141-4dc36180-6fa1-11eb-8645-4ffe06f6894d.PNG)

NEXT를 계속 눌러 설치를 완료해줍니다.

## STEP 4. cuDNN SDK 설치
[cuDNN SDK 다운로드 링크](https://developer.nvidia.com/cudnn)

Download **cuDNN을 눌러 Download** 받아줍니다.
![post9](https://user-images.githubusercontent.com/48666867/107912846-9cbdc680-6fa2-11eb-8b8a-7f436208826a.png)

멤버십이 요구 되므로, 회원가입 후 로그인을 진행합니다.<br>
자신이 설치한 CUDA 버전에 맞는 cuDNN을 선택하여 다운로드 합니다.

![post10](https://user-images.githubusercontent.com/48666867/107912888-b6f7a480-6fa2-11eb-9d80-ffbb6f972b12.png)
![post11](https://user-images.githubusercontent.com/48666867/107912977-e7d7d980-6fa2-11eb-830f-ebd4135756a1.png)

다운로드 받은 zip 파일의 압축을 해제합니다.

아래 그림과 같이 3개의 폴더가 있습니다. 안에 있는 파일을 CUDA Computing Toolkit에 복사합니다.

![post12](https://user-images.githubusercontent.com/48666867/107913200-4f8e2480-6fa3-11eb-8e53-2375cd609492.png)

cuda\bin 폴더 안의 모든 파일은 => C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin

cuda\include 폴더 안의 모든 파일은 => C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include

cuda\lib 폴더 안의 모든 파일은 => C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib

`Window + R` 키를 누른 후 `control sysdm.cpl`을 실행합니다.

![post13](https://user-images.githubusercontent.com/48666867/107913379-b01d6180-6fa3-11eb-96e0-086b0425ec9b.png)

**고급 탭 - 환경변수를 클릭**합니다.

![post14](https://user-images.githubusercontent.com/48666867/107913529-ea86fe80-6fa3-11eb-9a78-d5c6256606c0.png)

`CUDA_PATH`가 다음과 같이 잘 등록되어 있는지 확인합니다.
![post15](https://user-images.githubusercontent.com/48666867/107913626-1f935100-6fa4-11eb-8467-a956ecbbdbd4.PNG)
![post16](https://user-images.githubusercontent.com/48666867/107913646-25893200-6fa4-11eb-9718-df09335a44c0.PNG)

## Error
아래와 같이 에러가 났을 때 해결방안입니다.
![post17](https://user-images.githubusercontent.com/48666867/107922690-b9162f00-6fb3-11eb-87e0-9f789872ee06.jpg)

1. Move to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
2. Rename file `cusolver64_11.dll`  To  `cusolver64_10.dll`

<br><br>
## Reference
1. https://teddylee777.github.io/colab/tensorflow-gpu-install-windows
