---
layout: post
title: '[DeepLearning] CH05. 컴퓨터 비전을 위한 딥러닝(2)'
subtitle: 'hands-on DeepLearning'
categories: deeplearning
tags: deeplearning
comments: true
---
`케라스 창시자에게 배우는 딥러닝`을 기반으로 공부한 내용을 정리합니다.

<img src="/assets/img/dlcourse/book.jpeg" width="200" height="200">

## 5.2 소규모 데이터셋에서 밑바닥부터 컨브넷 훈련하기

 4,000개의 강아지와 고양이 사진(2,000개는 강아지, 2,000개는 고양이)으로 구성된 데이터셋에서 강아지와 고양이 이미지를 분류해보겠습니다.

 훈련을 위해 2,000개 사진을 사용하고 검증과 테스트에 각각 1,000개의 사진을 사용합니다.

 > 문제를 해결하기 위해 본 장에서 다루는 기본적인 전략이 있습니다.

 1) 보유한 2,000개의 소규모 데이터셋을 사용하여 처음부터 새로운 모델을 훈련하는 방법

 - 어떤 규제 방법도 사용하지 않고 훈련하여 기준이 되는 기본 성능을 만들어줍니다.
 - 71%의 분류 정확도를 달성하며, 과대적합 이슈가 발생합니다.

 2) **데이터 증식**(data augmentation)을 사용합니다.
 - 82% 정확도

 3) 사전 훈련된 네트워크의 특성을 이용하여 모델을 훈련합니다.
 - 특칭 추출기를 사용하여 90% 정확도
 - 세밀하기 튜닝(**fine-tuning**)을 통해 92% 정확도 달성

 ### 5.2.1 작은 데이터셋 문제에서 딥러닝의 타당성
 복잡한 문제를 푸는 컨브넷을 수십 개의 샘플만 사용해서 훈련하는 것은 불가능합니다. 하지만 모델이 작고 규제가 잘 되어 있으며 간단한 작업이라면 수백 개의 샘플로도 충분할 수 있습니다. 

 컨브넷은 지역적이고 평행 이동으로 변하지 않는 특성을 학습하기 때문에 지각에 관한 문제에서 매우 효율적으로 데이터를 사용합니다.

 매우 작은 이미지 데이터셋에서 어떤 종류의 특성 공학을 사용하지 않고 컨브넷을 처음부터 훈련해도 납득할 만한 결과를 만들 수 있습니다.

 또한 딥러닝 모델은 태생적으로 매우 다목적입니다. 대규모 데이터셋에서 훈련시킨 이미지 분류 모델이나 speech-to-text 모델을 조금만 변경해서 완전히 다른 문제에 재사용할 수 있습니다. <br>
 
 ### 5.2.2 데이터 내려받기
 [원본 데이터셋](https://www.kaggle.com/c/dogs-vs-catsdata)은 캐글에서 다운로드 받을 수 있습니다.

 아래 코드를 통해 각 클래스마다 1,000개의 샘플로 이루어진 훈련 세트, 500개 샘플의 검증 세트, 500개 샘플의 테스트 세트로 데이터셋을 만들어줍니다.

 ```python
 # code 5-4. Copy images to train, validation, test folder
 import os, shutil

# 원본 데이터셋을 압축 해제한 디렉터리 경로
original_dataset_dir = './datasets/cats_and_dogs/train'

# 소규모 데이터셋을 저장할 디렉터리
base_dir = './datasets/cats_and_dogs_small'
if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제
    shutil.rmtree(base_dir)   
os.mkdir(base_dir)

# 훈련, 검증, 테스트 분할을 위한 디렉터리
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 훈련용 고양이 사진 디렉터리
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# 훈련용 강아지 사진 디렉터리
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# 검증용 고양이 사진 디렉터리
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# 검증용 강아지 사진 디렉터리
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# 테스트용 고양이 사진 디렉터리
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# 테스트용 강아지 사진 디렉터리
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# 처음 1,000개의 고양이 이미지를 train_cats_dir에 복사
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 다음 500개 고양이 이미지를 validation_cats_dir에 복사
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# 다음 500개 고양이 이미지를 test_cats_dir에 복사
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# 처음 1,000개의 강아지 이미지를 train_dogs_dir에 복사
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 다음 500개 강아지 이미지를 validation_dogs_dir에 복사
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 다음 500개 강아지 이미지를 test_dogs_dir에 복사
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
```

이제 데이터 세트 준비가 완료되었습니다.

### 5.2.3 네트워크 구성하기
Conv2D(relu 활성화 함수 사용)와 MaxPooling2D 층을 번갈아 쌓은 컨브넷을 만들겠습니다. <br>
이전보다 이미지가 크고 복잡한 문제이므로 Conv2D + MaxPooling2D 단계를 추가하여 네트워크를 좀 더 크게 만들어줍니다.<br>
이렇게하면 네트워크의 용량을 늘리고 Flatten 층의 크기가 너무 커지지 않도록 특성 맵의 크기를 줄일 수 있습니다. <br>
150 x 150 크기(임의 선택)의 입력으로 시작해 Flatten 층 이전에 7 x 7 크기의 특성 맵으로 줄어듭니다.

> 특성 맵의 깊이는 네트워크에서 점진적으로 증가하지만(32 -> 128), 특성 맵의 크기는 감소합니다(150 x 150 -> 7 x 7). 거의 모든 컨브넷에서 볼 수 있는 전형적인 패턴입니다.

이진 분류 문제이므로 네트워크는 하나의 유닛(크기가 1인 Dense 층)과 sigmoid 활성화 함수로 끝납니다. 이 유닛은 한 클래스에 대한 확률을 인코딩할 것입니다.

```python
# code 5-5 Create small ConvNet for CAT vs DOG Classification
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten()) # 구분기의 입력으로 연결하기 위하여 3D 텐서를 1D 텐서로 펼침
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

![img](/assets/img/dlcourse/summary04.png)

컴파일 단계에서 RMSprop 옵티마이저를 선택하겠습니다. 네트워크의 마지막이 하나의 시그모이드 유닛이기 때문에 이진 크로스엔트로피(binary crossentropy)를 손실로 사용합니다.

```python
# code 5-6 Set train model
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

손실 함수 목록을 참고하면 됩니다.
![img](/assets/img/dlcourse/capture12.png)

### 5.2.4 데이터 전처리
데이터는 네트워크에 주입되기 전에 부동 소수 타입의 텐서로 전처리되어 있어야 합니다. <br>
지금은 데이터가 JPEG 파일로 되어 있으므로 다음 과정을 따라줍니다.

1. 사진 파일을 읽어줍니다.
2. JPEG 콘텐츠를 RGB 픽셀 값으로 디코딩합니다.
3. 그 다음 부동 소수 타입의 텐서로 변환합니다.
4. 픽셀 값(0 ~ 255)의 스케일을 [0, 1] 사이로 조정합니다(신경마은 작은 입력 값을 선호).

케라스에는 `keras.preprocessing.image`에 이미지 처리를 위한 헬퍼 도구가 있습니다(위 과정을 자동으로 처리하는 유틸리티). <br>
특히 `ImageDataGenerator` 클래스는 디스크에 있는 이미지 파일을 전처리된 배치 텐서로 자동으로 바꾸어 주는 파이썬 제너레이터(generator)를 만들어 줍니다.

```python
# code 5-7 Read Image on Directory used to ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

# 모든 이미지를 1/255로 스케일 조정
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 타깃 디렉터리
        train_dir,
        # 모든 이미지를 150 × 150 크기로 변경
        target_size=(150,150),
        batch_size=20,
        # binary_crossentropy 손실을 사용하기 때문에 이진 레이블 필요
        # (다중분류시 class_mode='categorical'(원-핫 인코딩 레이블) 혹은 class_mode='sparse'(정수형 레이블))
        class_mode='binary'
        )

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
        )
```

제너레이터의 출력은 150 X 150 RGB 이미지의 배치((20, 150, 150, 3)크기)와 이진 레이블의 배치((20,)크기)입니다. <br>
각 배치에는 20개의 샘플(batch size)이 있습니다. <br>
제너레이터는 이 배치를 무한정 만들어내기 때문에 타깃 폴더에 있는 이미지를 끝없이 반복합니다. 따라서 반복 루프 안 어디에선가 `break`문을 사용해야 합니다.

```python
for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape)
    print('배치 레이블 크기:', labels_batch.shape)
    break
배치 데이터 크기: (20, 150, 150, 3)
배치 레이블 크기: (20,)
```
이제 제너레이터를 사용한 데이터에 모델을 훈련시키겠습니다. <br>
`fit_generator` 메소드는 `fit` 메소드와 동일하되 데이터 제너레이터를 사용할 수 있으며 첫 번째 매개변수로 입력과 타깃의 배치를 끝없이 반환하는 파이썬 제너레이터를 기대합니다.

데이터가 끝없이 생성되기 때문에 케라스 모델에 하나의 에포크를 정의하기 위해 제너레이터로부터 얼마나 많은 샘플을 뽑을 것인지 `steps_per_epoch` 매개변수를 통해 알려줍니다. <br>
제너레이터로부터 `steps_per_epoch`개의 배치만큼 뽑은 후, 즉 `steps_per_epoch` 횟수만큼 경사 하강법 단계를 실행 한 후 훈련 프로세스는 다음 에포크로 넘어갑니다.

20개의 샘플이 하나의 배치이므로 2,000개의 샘플을 모두 처리할 때까지 100개의 배치를 뽑을 것입니다.

`fit_generator`를 사용할 때 `validation_data` 매개변수를 전달할 수 있습니다. <br>
이 매개변수에 데이터 제너레이터도 가능하지만 넘파이 배열의 튜플도 가능합니다.

`validation_data`로 제너레이터를 전달하면 검증 데이터의 배치를 끝없이 반화합니다. 따라서 `validation_steps` 매개변수에 얼마나 많은 배치를 추출하여 평가할지 지정해줍니다.

```python
# code 5-8 Train Model used to Batch Generator
# 제너레이터 사용시 fit_generator 메서드 사용
history = model.fit_generator(
      train_generator, #입력과 레이블의 배치를 끝없이 생성
      steps_per_epoch=100, #20개의 샘플이 하나의 배치, 2,000개 샘플을 모두 처리할때까지 100개의 배치 추출이 필요
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50) #배치가 20개로 설정, 전체 검증 데이터 1,000개를 사용하려면, 50개의 배치 추출 필요
```

```python
# code 5-9 Save Model
model.save('cats_and_dogs_small_1.h5')
```

이제 훈련 데이터와 검증 데이터에 대한 모델의 손실과 정확도를 그래프로 나타내 보겠습니다.

```python
# code 5-10 Show Graph to Accuracy and Loss
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
![img](/assets/img/dlcourse/accuracy01.png)
![img](/assets/img/dlcourse/loss01.png)

위 그래프는 과대적합의 특성을 보여줍니다. <br>
훈련 정확도가 선형적으로 증가해서 거의 100% 도달하는 반면 검증 정확도는 70~72%에서 정체되었습니다. <br>
검증 손실은 다섯 번의 에포크만에 최솟값에 다다른 이후 더 이상 진전되지 않으며 훈련 손실은 거의 0에 도달할 때까지 선형적으로 계속 감소합니다.

드롭아웃이나 가중치 감소(L2 규제)처럼 과대적합을 감소시킬 수 있는 여러 가지 기법들을 적용할 수 있습니다. <br>
본 장에선 컴퓨터 비전에 특화되어 있어서 딥러닝으로 이미지를 다룰 때 흔히 사용되는 **데이터 증식**을 사용해보겠습니다.

### 5.2.5 데이터 증식 사용하기
과대적합은 학습할 샘플이 너무 적어 새로운 데이터에 일반화할 수 있는 모델을 훈련시킬 수 없기 때문에 발생합니다.<br>
무난히 많은 데이터가 주어지면 데이터 분포의 모든 가능한 측면을 모델이 학습할 수 있습니다.

데이터 증식은 기존 훈련 샘플로부터 더 많은 훈련 데이터를 생성하는 방법입니다. 그럴듯한 이미지를 생성하도록 여러 가지 랜덤한 변환을 적용하여 샘플을 늘립니다.

훈련할 때 모델이 정확히 같은 데이터를 두 번 만나지 않도록 하는 것이 목표입니다. 모델이 데이터의 여러 측면을 학습하면 **일반화**에 도움이 될 것입니다.

케라스에서는 `ImageDataGenerator`가 읽은 이미지에 여러 종류의 랜덤 변환을 적용하도록 설정할 수 있습니다.

```python
# code 5-11 Set Data Augmentation used to ImageDataGenerator
# ImageDataGenerator를 상용하여 데이터 증식 설정하기
datagen = ImageDataGenerator(
      rotation_range=40,  # 랜덤하게 이미지를 회전시킬 각도 범위 설정
      width_shift_range=0.2,  # 이미지를 수평으로 랜덤하게 평행이동 시킬 범위 설정
      height_shift_range=0.2, # 이미지를 수직으로 랜덤하게 평행이동 시킬 범위 설정
      shear_range=0.2, # 랜덤하게 전단변환(shearing transforming)을 적용할 각도 범위 설정
      zoom_range=0.2, # 랜덤하게 사진을 확대할 범위 설정
      horizontal_flip=True, # 랜덤하게 이미지를 수평으로 뒤집음
      fill_mode='nearest') # 회전이나 가로/세로 이동으로 인하여 새롭게 생성해야 할 픽셀을 채우는 방법 설정
```
> 매개 변수 옵션
- `rotation_range`: 랜덤하게 사진을 회전시킬 각도 범위(0~180)
- `width_shift_range`와 `height_shift_range`: 사진을 수평, 수직으로 랜덤하게 평행 이동시킬 범위(전체 너비와 높이에 대한 비율)
- `shear_range`: 랜덤하게 전단 변환(shearing transformation)을 적용할 각도 범위
- `zoom_range`: 랜덤하게 사진을 확대할 범위
- `horizontal_flip`: 랜덤하게 이미지를 수평으로 뒤집습니다. 수평 대칭을 가정할 수 있을 떄 사용(ex. 풍경/인물 사진)
- `fill_mode`: 회전이나 가로/세로 이동으로 인해 새롭게 생성해야 할 픽셀을 채울 전략

[전체 매개변수](https://keras.io/api/preprocessing/image/)에 대한 설명은 하이퍼링크를 통해 확인할 수 있습니다.

증식된 이미지 샘플을 살펴보겠습니다.

```python
# code 5-12 Show Randomly Augmentative Train image 
# 이미지 전처리 유틸리티 모듈
from keras.preprocessing import image

fnames = sorted([os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)])

# 증식할 이미지 선택
img_path = fnames[3]

# 이미지를 읽고 크기를 변경
img = image.load_img(img_path, target_size=(150, 150))

# (150, 150, 3) 크기의 넘파이 배열로 변환
x = image.img_to_array(img)

# (1, 150, 150, 3) 크기로 변환
x = x.reshape((1,) + x.shape)

# flow() 메서드는 랜덤하게 변환된 이미지의 배치를 생성,
# 무한 반복되기 때문에 어느 지점에서 중지필요
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()
```
![img](/assets/img/dlcourse/capture13.png)

데이터 증식을 사용하여 새로운 네트워크를 훈련시킬 때 네트워크에 **같은 입력 데이터가 두번 주입되지 않습니다**. 하지만 적은 수의 원본 이미지에서 만들어졌기 때문에 입력 데이터들 사이에 상호 연관성이 큽니다.<br>
즉 새로운 정보를 만들어 낼 수 없고 단지 기존 정보의 재조합만 가능합니다. <br>
따라서 완전히 과대적합을 제거하기에 충분하지 않을 수 있어 완전 연결 분류기 직전에 `Dropout` 층을 추가하여 과대적합을 더 억제하겠습니다.

```python
# code 5-13 Define new ConvNet including Dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())     # 구분기의 입력으로 연결하기 위하여 3D 텐서를 1D 텐서로 펼침
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
```

네트워크를 훈련시켜 줍니다.

```python
# code 5-14 Train ConvNet used to Data Augmentation Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# 검증 데이터는 증식되어서는 안 됨
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 타깃 디렉터리
        train_dir,
        # 모든 이미지를 150 × 150 크기로 변경
        target_size=(150, 150),
        batch_size=32,
        # binary_crossentropy 손실을 사용하기 때문에 이진 레이블을 만듬
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data = validation_generator,
      validation_steps=50)
```

모델을 저장해줍니다.
```python
# code 5-15 Save Model
model.save('cats_and_dogs_small_2.h5')
```

데이터 증식과 드롭아웃 덕분에 더 이상 과대적합되지 않습니다. <br>
훈련 곡선이 검증 곡선에 가깝게 따라가고 있으며 검증 데이터에서 82% 정확도를 달성하였습니다. <br>
규제하지 않은 모델과 비교했을 때 15% 정도 향상되었습니다.

![img](/assets/img/dlcourse/accuracy02.png)
![img](/assets/img/dlcourse/loss02.png)


<br><br>

## Reference
1. 케라스 창시자에게 배우는 딥러닝
2. https://www.kaggle.com/
