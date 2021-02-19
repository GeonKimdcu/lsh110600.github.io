---
layout: post
title: '[ArcFault] Project step01: Data Collection'
subtitle: 'data collection'
categories: sideproject
tags: arcfault
comments: true
---
`Arc-Fault` 연구과제를 위해 개발한 코드입니다.

## Introduction
[Github](https://github.com/GeonKimdcu/SideProject) 들어가시면 프로젝트 관련 상세 내용 및 전체 코드가 있습니다. <br><br>
etri 동계 연구 연수생으로 활동하게 되면서 박사님 밑에서 실제 진행 중이신 연구과제를 보조하게 되었습니다. 이 프로젝트의 코드는 본 연구과제에 적용하기 위해 직접 신호처리에 대해 study하며 개발한 코드입니다. <br>

이번 포스트에서는 먼저 데이터 수집 단계에 대해서 다뤄보겠습니다. <br>
박사님께서 request한 사항들을 먼저 정리한 후 거기에 맞춰 코드 구현을 진행하겠습니다.

## Requests
- 실제 연구과제에 적용될 데이터 수집 일정이 미뤄졌습니다. 따라서 데이터가 없는 관계로 실제 데이터와 비슷한 값을 가진 가상의 데이터를 생성하였습니다.
- 데이터의 종류는 3종(정상상태, Arc1, Arc2)이며 각 종류 당 20개의 데이터를 추출해야합니다.
- signal 중 Zero-Crossing Detector 부분만 추출하여 feature을 뽑아내 종류별로 20개의 데이터를 수집합니다.
- Data Freatures:
    - 신호의 최대값, 평균값, 표준편차, 최대값과 최솟값의 차
    - FFT(Fast Fourier Transform) 취한 후 주파수의 최대값, 두 번째로 큰 최대값, Magnitude의 최대값, 두 번째로 큰 최대값
    - STFT(Short Time Fourier Transform) 취한 후 windows 중 주파수 최대값, 최대값 중 가장 작은 값, 주파수 변화율, windows 중 Magnitude의 최대값, 최대값 중 가장 작은 값

## Generate Signal
우선 신호를 생성할 때 박사님께서 요구하신 조건에 맞추어 생성해주었습니다.<br>
sampling rate(sampling frequency)는 10MHz이며, 주파수는 60Hz, amplitude는 2인 신호를 생성하였습니다. 거기에 랜덤한 노이즈를 추가하기 위해 난수를 이용하여 랜덤한 주파수 1MHz ~ 1.5MHz, amplitude는 0.2인 신호를 생성 후 둘의 신호를 합성해주었습니다.
```python
# sampling rate
fs = 10000000 # 10MHz

# signal length
t = np.arange(0, 0.035, 1 / fs) # s, sampling interval, time array

# generate signal
f1 = 60 # 60Hz
signal_f = 2*np.sin(2*np.pi*f1*t)  # amplitude = 2

# generate noise signal
np.random.seed(222)
n1 = np.random.uniform(1000000, 1500000, len(t)) # 1MHz ~ 1.5 MHz
signal_n = 0.2*np.sin(2*np.pi*n1*t)

# total signal
normal_signal = signal_f + signal_n
```
생성한 신호를 시각화해보겠습니다.

```python
# visualizing signal

plt.figure(num = 1, dpi = 100)
plt.plot(t, normal_signal)
plt.grid()
```
![arcpost06](https://user-images.githubusercontent.com/48666867/106717356-f6d59800-6642-11eb-978e-8542e3ccfce5.PNG)

이제 Arc1과 Arc2의 신호를 생성해보겠습니다.

```python
#  generate noise Arc 1 signal

n2 = 3000000 # 3MHz
np.random.seed(222)
random_mag_Arc1 = np.random.uniform(0.5, 0.8, 1500)
signal_Arc1 = random_mag_Arc1*np.sin(2*np.pi*n2*t[82500:84000]) # random magnitude 0.5 ~ 1

# total signal
ext_signal_Arc1 = normal_signal[82500:84000] + signal_Arc1
```
Arc1 신호는 magnitude의 범위를 0.5 ~ 0.8로 랜덤하게 설정해줍니다. 주파수는 3MHz로 해줍니다. 82500:84000은 앞서 생성한 신호 중 Zero-Crossing Detection의 범위입니다.
실제 Arc 신호는 Zero-Crossing Detection에서 발생하기 때문에 위와 같이 범위를 설정해줍니다.

```python
#  generate noise Arc 2 signal

n3 = 4000000 # 4MHz

np.random.seed(333)
random_mag_Arc2 = np.random.uniform(0.8, 1.1, 1500)
signal_Arc2 = random_mag_Arc2*np.sin(2*np.pi*n3*t[82500:84000])

# total signal
ext_signal_Arc2 = normal_signal[82500:84000] + signal_Arc2
```
Arc2 신호는 magnitude의 범위를 0.8 ~ 1.1로 랜덤하게 설정하고, 주파수는 4MHz로 해줍니다.

<br>


## Extracted Signal
이제 Normal, Arc1, Arc2 state 별 위와 같은 범위의 신호를 추출하여 살펴보겠습니다.

```python
# Normal state
ext_t = t[82500:84000]
ext_signal = normal_signal[82500:84000]

plt.figure(num = 1, dpi = 100)
plt.plot(ext_t, ext_signal)
plt.grid()
```
![arcpost07](https://user-images.githubusercontent.com/48666867/106720008-4a95b080-6646-11eb-82a9-70738b1d18b4.PNG)

```python
# Arc1 state
plt.figure(num = 1, dpi = 100)
plt.plot(ext_signal_Arc1)
plt.grid()
```
![arcpost08](https://user-images.githubusercontent.com/48666867/106720224-8df01f00-6646-11eb-9867-76f086ae0d7c.PNG)

```python
# Arc2 state
plt.figure(num = 1, dpi = 100)
plt.plot(ext_signal_Arc2)
plt.grid()
```
![arcpost09](https://user-images.githubusercontent.com/48666867/106720600-01922c00-6647-11eb-90d3-ea5be4178349.PNG)

## Compute the Fast Fourier Transform
추출한 신호를 FFT 적용하여 각 상태 별 주파수의 분포를 확인해보겠습니다.

FFT 변환을 해주면 대칭된 분포를 나타내기 때문에 필요한 범위만 나타내도록 Y값의 범위를 n/2로 해줍니다. <br>
y축은 amplitude를 의미하고, x축은 주파수를 의미합니다. <br>

normal state signal의 FFT 변환 결과 입니다.
출력 결과를 보시면 많은 노이즈가 껴있는 것을 확인 할 수 있습니다.
```python
n = len(ext_signal)
f = np.linspace(0,fs/2, math.trunc(n/2))

Y = np.fft.fft(ext_signal) / n
Y = Y[range(math.trunc(n/2))]
amplitude_Hz = 2*abs(Y)

plt.stem(f, amplitude_Hz)
```
![arcpost10](https://user-images.githubusercontent.com/48666867/106721761-6dc15f80-6648-11eb-959c-bd87474ba62a.PNG)

Arc1 state signal의 FFT 변환 결과입니다.<br>
Arc1의 주파수인 3MHz와 0.5 ~ 0.8로 주었던 amplitude가 잘 나온 것을 확인할 수 있습니다.

```python
n = len(ext_signal_Arc1)
f = np.linspace(0, fs/2, math.trunc(n/2))

Y = np.fft.fft(ext_signal_Arc1) / n
Y = Y[range(math.trunc(n/2))]
amplitude_Hz = 2*abs(Y)

plt.stem(f, amplitude_Hz)
```
![arcpost11](https://user-images.githubusercontent.com/48666867/106722583-7cf4dd00-6649-11eb-8686-3bf41a2134c5.PNG)

Arc2 state signal의 FFT 변환 결과입니다.<br>
Arc2의 주파수인 4MHz와 0.8 ~ 1.1로 주었던 amplitude가 잘 나온 것을 확인할 수 있습니다.
```python
n = len(ext_signal_Arc2)
f = np.linspace(0, fs/2, math.trunc(n/2))

Y = np.fft.fft(ext_signal_Arc2) / n
Y = Y[range(math.trunc(n/2))]
amplitude_Hz = 2*abs(Y)

plt.stem(f, amplitude_Hz)
```
![arcpost15](https://user-images.githubusercontent.com/48666867/106831328-0ef4f800-66d3-11eb-89c7-d48d655bbf78.PNG)

## Compute the Short Time Fourier Transform
다음은 시간에 따른 주파수 성분의 변화를 파악하기 위해 STFT를 적용해보겠습니다. <br>
`signal.stft`함수를 사용하여 return 값으로 f, t, Zxx를 취해줍니다. 
- f: 주파수(Hz)array
- t: 시간 축(Time array)
- Zxx: 시간에 따른 주파수 세기
- 함수 argument: 1500(signal length) / 300(nperseg) = 5개의 window로 division
- nfft: segment 길이 300이므로 2^n을 취하여 2^9 = 512로 더 크게 잡아줍니다. (2^8=256 < 300)
```python
f, t, Zxx = signal.stft(ext_signal, fs = fs, nperseg = 300, nfft = 512)
plt.pcolormesh(t, f, np.abs(Zxx), shading = 'gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
```
![arcpost12](https://user-images.githubusercontent.com/48666867/106827851-8b380d00-66cc-11eb-918f-fab36344c059.PNG)

Normal state이기 때문에 특정하게 강력한 주파수는 없고, 많은 노이즈들이 보이는 것을 알 수 있습니다.

다음으로 Arc1 state를 살펴보겠습니다.
```python
f_axis, t_axis, Zxx = signal.stft(ext_signal_Arc1, fs = fs, nperseg = 300)
plt.pcolormesh(t_axis, f_axis, np.abs(Zxx), shading = 'gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
```
![arcpost13](https://user-images.githubusercontent.com/48666867/106828278-38ab2080-66cd-11eb-9ee6-295476975d3b.PNG)

Arc1 신호의 주파수인 3MHz가 시간에 상관없이 일정하게 나타나는 것을 볼 수 있습니다.
다음으로 Arc2 state를 살펴보겠습니다.
```python
f_axis, t_axis, Zxx = signal.stft(ext_signal_Arc2, fs = fs)
plt.pcolormesh(t_axis, f_axis, np.abs(Zxx), shading = 'gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
```
![arcpost14](https://user-images.githubusercontent.com/48666867/106828580-d7d01800-66cd-11eb-8aa5-22bc18a17d12.PNG)

마찬가지로 Arc2 신호의 주파수인 4MHz가 시간에 상관없이 일정하게 나타나는 것을 볼 수 있습니다.<br>

## Data Collection
지금까지 하나의 기존 신호에 Zero-Crossing Detection 추출하여 그 신호를 가지고 여러 transform을 취해보았습니다. 이제 실제 데이터와 비슷하게 랜덤한 기존 신호를 여러개 생성하여 각각의 feature들을 수집해보겠습니다.
<br>
<br>
우선 STFT 변환 후 feature들을 뽑아 낼 수 있는 함수를 정의해보겠습니다.
```python
def stft_signal(state_signal, fs):
    
    # Compute the Short Time Fourier Transform (STFT)
    f_axis, t_axis, Zxx = signal.stft(state_signal, fs = fs, nperseg = 300, nfft = 512)
    
    # 5 window division ...... f0 ~ f4로 나눠줌
    for i, j in enumerate(np.arange(0, len(t_axis)-2, 2)):
        globals()['f{0}'.format(i)] = abs(Zxx[:,j:3+j])
        
    # Max Magnitude of f0 ~ f4 ...... f0 ~ f4중에서 가장 큰 Magnitude값을 찾음.
    Max_mag = f0.max()
    
    for i in range(1, 5):
        a = globals()['f{0}'.format(i)]
        
        if Max_mag < a.max():
            Max_mag = a.max()
            
    # Max Hz of f0 ~ f4  ... f0~f4 중에서 가장 큰 Magnitude 값을 가진 주파수를 찾음.
    HZ = abs(Zxx)
    
    X, Y = np.where(HZ == Max_mag)
    Max_Hz_index = X[0]
    Max_Hz = f_axis[Max_Hz_index]
    
    # Min Magnitude of f0 ~ f4 ...... f0 ~ f4의 Max값들 중 가장 작은 Magnitude값을 찾음.
    Min_mag = f0.max()
    
    for i in range(1, 5):
        a = globals()['f{0}'.format(i)]
        # print(a.max())
        
        if Min_mag > a.max():
            Min_mag = a.max()
    
    # Min Hz of f0 ~ f4......f0~f4의 Max값들 중 가장 작은 Magnitude 값을 가진 주파수를 찾음.
    X, Y = np.where(HZ == Min_mag)
    Min_Hz_index = X[0]
    Min_Hz = f_axis[Min_Hz_index]
    
    # Max Hz - Min Hz = 변화율
    Hz_diff = Max_Hz - Min_Hz
    
    return (Max_mag, Min_mag, Max_Hz, Min_Hz, Hz_diff)
```

<br>
그리고 데이터의 종류(normal, Arc1, Arc2) 별로 각각의 데이터를 수집해주는 함수를 정의해보겠습니다. 대표적으로 normal 함수만 살펴보겠습니다.

```python
normal_Max = []
normal_Mean = []
normal_Std = []
normal_Full_diff = []
normal_FFT_Hz_Max1 = []
normal_FFT_Mag_Max1 = []
normal_FFT_Hz_Max2 = []
normal_FFT_Mag_Max2 = []
normal_STFT_Hz_Max = []
normal_STFT_Hz_Min = []
normal_STFT_Mag_Max = []
normal_STFT_Mag_Min = []
normal_STFT_Hz_diff = []

def normal_signal(i):
    # sampling rate
    fs = 10000000 # 10MHz
    
    # signal length
    t = np.arange(0, 0.035, 1 / fs) # s, sampling interval, time array

    # generate signal
    f1 = 60 # 60Hz
    signal_f = 2*np.sin(2*np.pi*f1*t)  # amplitude = 2

    # generate noise signal
    np.random.seed(i)
    n1 = np.random.uniform(1000000, 1500000, len(t)) # 1MHz ~ 1.5 MHz
    signal_n = 0.2*np.sin(2*np.pi*n1*t)

    # join signal
    n_signal = signal_f + signal_n
    
    # extract signal
    ext_signal = n_signal[82500:84000] # shoulder 부분 추출
    
    # Fourier transformed Normal signal
    n = 2048 # len(ext_signal) < 2^11 = 2048
    f = np.linspace(1000000, 1500000, math.trunc(n/2)) # 주파수 영역 값 범위 설정
    
    Y = np.fft.fft(ext_signal) / n # fft 변환 값 Y에 대입
    Y = Y[range(math.trunc(n/2))] # 대칭이므로 뒷부분 생략
    amplitude_Hz = 2*abs(Y) # 절대값 취한 후 2배해서 크기 나타냄
    
    # Short-time Fourier transformed Normal signal
    Max_mag, Min_mag, Max_Hz, Min_Hz, Hz_diff = stft_signal(ext_signal, 10000000)

    # collect data
    normal_Max.append(ext_signal.max()) # Max 값 feature 저장 
    normal_Mean.append(ext_signal.mean()) # Mean 값 feature 저장
    normal_Std.append(ext_signal.std()) # 표준편차 feature 저장
    normal_Full_diff.append(ext_signal.max() - ext_signal.min()) # Max 값 - Min 값 차이
    normal_FFT_Mag_Max1.append(amplitude_Hz.max()) # FFT 변환 후 가장 큰 Magnitude
    normal_FFT_Mag_Max2.append(sorted(amplitude_Hz, reverse = True)[1]) # 두 번쨰로 가장 큰 Magnitude
    normal_FFT_Hz_Max1.append(f[amplitude_Hz.argmax()]) # 가장 큰 주파수 값
    
    # 두 번째로 큰 주파수 값을 찾기 위한 임시 변수 생성 (test)
    test = np.where(amplitude_Hz == sorted(amplitude_Hz, reverse = True)[1])
    test = list(test)
    normal_FFT_Hz_Max2.append(f[test[0][0]]) # 두 번째로 큰 주파수 값
    
    # STFT 변환 함수 반환값들 저장
    normal_STFT_Mag_Max.append(Max_mag)
    normal_STFT_Mag_Min.append(Min_mag)
    normal_STFT_Hz_Max.append(Max_Hz)
    normal_STFT_Hz_Min.append(Min_Hz)
    normal_STFT_Hz_diff.append(Hz_diff)
```

데이터 feature 수집 함수를 생성하였으면, 이제 신호 1개마다 feature를 뽑아 list에 저장하고 이를 DataFrame으로 변환해줍니다.

```python
# update seed

for i in range(0, 20):
    normal_signal(i)
    
# create DataFrame
normal_state = pd.DataFrame({"Max" : normal_Max,
                            "Mean" : normal_Mean,
                            "Std" : normal_Std,
                            "Full_diff": normal_Full_diff,
                            "FFT_Mag1" : normal_FFT_Mag_Max1,
                            "FFT_Hz1" : normal_FFT_Hz_Max1,
                            "FFT_Mag2" : normal_FFT_Mag_Max2,
                            "FFT_Hz2" : normal_FFT_Hz_Max2,
                            "STFT_Mag1" : normal_STFT_Mag_Max,
                            "STFT_Hz1" : normal_STFT_Hz_Max,
                            "STFT_Mag2" : normal_STFT_Mag_Min,
                            "STFT_Hz2" : normal_STFT_Hz_Min,
                            "STFT_diff" : normal_STFT_Hz_diff})
```
이로써 normal state의 데이터의 수집을 완료했습니다.
마찬가지로 Arc1, Arc2도 똑같은 과정을 거쳐 총 row가 60인 데이터 수집이 완료됩니다.
<br>
<br>
<br>
이상 Arc-Fault project에서 데이터 수집단계에 대해 코드 구현이 어떻게 이루어지는지 살펴보았습니다. 만약 코드 내용이 잘 이해되지 않는다면 앞선 [포스트](https://geonkimdcu.github.io/sideproject/2021/02/01/SP-ArcFault-0/)에서 Transform에 대한 기초적인 내용을 숙지하고 보시면 될 것 같습니다.  <br>
다음 포스트는 수집한 데이터를 이용한 분류 모델에 대해 업로드 하도록 하겠습니다.

감사합니다 :)