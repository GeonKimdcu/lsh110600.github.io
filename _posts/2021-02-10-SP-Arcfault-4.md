title: '[ArcFault] Project step03: ArcFault Detection'
subtitle: 'data analysis'
categories: sideproject
tags: arcfault
comments: true
---
`Arc-Fault` 연구과제를 위해 개발한 코드입니다.

## Introduction
저번 [포스트](https://geonkimdcu.github.io/sideproject/2021/02/09/SP-ArcFault-3/)에선 데이터 분석 단계에 대해 알아보았습니다. <br>
이번 포스팅은 시계열 데이터를 통해 Arc 상태를 탐지해보는 시간을 갖겠습니다. <br><br>
전체 코드는 제 [Github](https://github.com/GeonKimdcu/SideProject)에 업로드 되어 있습니다.

## Generate Arc signal
우선 아크 신호를 생성해줍니다.
```python
# sampling rate
fs = 10000000 # 10MHz

# signal length
# s, sampling interval, time array
t = np.arange(0, 0.5, 1 / fs)

# generate signal
f1 = 60 # 60Hz
signal_f = 2*np.sin(2*np.pi*f1*t)  # amplitude = 2

# generate noise signal
np.random.seed(222)
n1 = np.random.uniform(1000000, 1500000, len(t)) # 1MHz ~ 1.5 MHz
signal_n = 0.2*np.sin(2*np.pi*n1*t)

# total signal
normal_signal = signal_f + signal_n

#  generate noise Arc signal

n3 = 4000000 # 4MHz

np.random.seed(333)
random_mag_Arc = np.random.uniform(0.8, 1.1, 1500)
signal_Arc = random_mag_Arc*np.sin(2*np.pi*n3*t[82500:84000])

signal_Arc # 아크 신호

# shoulder 범위만 아크 신호를 더해주기 위해 타입변환
temp = list(normal_signal)

# shoulder 범위만 아크 신호 더하고 shoulder 범위가 아닌 부분은 아크신호를 더하지 않음.

# shoulder 범위에 아크신호를 더해주기 위해 범위 설정
i,j = 1, 0
s0 = temp[82500*0+1500*0:82500*1+1500*0]
end = (82500*1)+(1500*0)

# 신호 길이(데이터 개수) 끝까지 반복해서 noise 섞인 sin파 신호와 아크 신호 합성을 반복해줌.
while (len(s0) != 5000000):
    start = end
    np.random.seed(0)
    # 아크 신호를 랜덤하기 더해주기 위한 규칙 생성
    num = random.randint(0,2)
    
    if i == j: # shoulder 범위가 아닐때 그냥 신호 더해줌.
        i += 1
        end = (82500*i)+(1500*j)
        s0 = s0+temp[start:end]
    # shoulder 범위 이면서 num == 2일 경우 아크 신호 더함.
    elif i!=j and num == 2:
        j += 1
        end = (82500*i)+(1500*j)
        x = temp[start:end] + signal_Arc
        s0 = s0+ list(x) 
    else: # shoulder 범위 이나 num은 2가 아닐 경우 노말 신호만 더해줌.
        j += 1
        end = (82500*i)+(1500*j)
        s0 = s0+ temp[start:end] 

# numpy array type으로 변환, 아크 신호가 합쳐진 signal 데이터 완성        
sig = np.array(s0)
```

## visualizing signal

노말과 아크 상태를 추가하여 완성된 신호를 시각화해줍니다.

plt.figure(num = 1, dpi = 100)
plt.plot(t, sig)
plt.grid()

![img](/assets/img/arcpost/arc_picture01.png)

## Arc Fault Detection
아크상태를 실시간으로 검출해주는 코드입니다. 화재 위험 단계를 `safety`, `caution`, `warning` 3가지로 구분하여 출력합니다.
`global count`값이 설정해둔 `threshold`를 초과하면 단계별로 출력하는 코드입니다.

```python
# ratio 계수에 윈도우 shift하면서 중복된 값을 제거해주며, 순서는 그대로 유지해주는 함수 생성
def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

# Wavelet Packet Transform 변환 후 시각화 함수
def WPT(signal):
    wp = pywt.WaveletPacket(data = signal, wavelet = 'db10', mode = 'symmetric') # 'db10'의 wavelet family 사용
    xx = wp['ddd'].data # 'ddd'(3 level)에서 고주파 성분 추출
    
    # 논문 I_ratio 수식 인용
    shift_size = 6000 #1/4 cycle
    window_length = 12000 # 1/2 cycle
    ratio = []
    I_max = xx[:12000].max() # 첫 번째 window max값(고정)
    
    # window shift
    for i in np.arange(0, len(xx), 6000):
        if len(ratio) < 104:
            window_2 = xx[i + shift_size: i + shift_size + window_length]

            I_n_max = window_2.max()

            I_ratio = I_n_max / I_max

            ratio.append(I_ratio)
            
    ratio= unique(ratio) # ratio 중복 값 제거
    tt = np.linspace(0, 0.5, len(ratio))
    
    fig = plt.figure(figsize = (15, 6))
    # 그래프 2개 출력
    spec = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[1, 3])
    ax0 = fig.add_subplot(spec[0])
    ax0.plot(xx)
    
    ax1 = fig.add_subplot(spec[1])
    ax1.scatter(tt, ratio)
    ax1.plot(tt, ratio, 'yx-')
    plt.axhline(y = 1.6, color = 'r', linewidth = 2) # warning threshold
    plt.axhline(y = 1.2, color = 'g', linewidth = 2) # attention threshold
    
    global count
    count = 0
    
    # count 개수에 따라 현재 상태 출력
    def state(count):
        if count > 2:
            print("Warning!")
        elif count > 1:
            print("Attention!")
        elif 0 <= count < 2:
            print("Safety")
        elif count < 0: # 이 부분 빼고 위에 위험도 단계별 count 기준 수정 필요.
            count = 0
            print("Safety")
    
    for i in ratio:
        if i > 1.6: # warning threshold 값 넣어주면 됨.
            count += 1
            state(count)
        elif 1.2 < i < 1.6: # attention threshold 값 넣어주면 됨.
            state(count)
        else:
            if count < 0:
                count= 0
                state(count)
            else:
                count -= 1
                state(count)
    
WPT(sig)
```
그러면 해당 상태를 출력합니다.

Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Attention!
Safety
Attention!
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Attention!
Warning!
Warning!
Warning!
Attention!
Safety
Attention!
Warning!
Attention!
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety
Safety

아래의 사진은 정상신호와 아크신호를 결합한 신호를 보여주며,  초록선은 주의 단계, 빨간선은 위험단계 임계값을 설정해준 것입니다. 이 임계값을 넘으면 카운트가 누적되며, 현 상황에 대한 화재 위험 단계를 실시간으로 출력해줍니다.

![img](/assets/img/arcpost/arc_picture02.png)

<br><br>

## 연구를 끝마치면서..
2개월 동안 본 연구를 진행하면서 신호처리에 대해 정말 깊이 배운 것 같습니다. 
<br>
제가 진행한 활동은 연구를 시작하기에 앞서 가상의 데이터를 생성하여 프레임을 제작한 것이므로 실제 데이터를 수집하게 된 후 해당 코드에 input하면 어떻게 될지는 모르겠습니다. 실제 데이터로 본 코드에 적용해보지 못해 아쉽습니다. 그리고 본 결과물에서 더욱 수정&보완된 최종 연구 결과물이 궁금해집니다.


## Reference
1. H.Zhang, "Arc Fault Signatures Detection on Aircraft Wiring System", IEEE, 2006, pp. 5548-5552
