---
layout: post
title: '[Docker] Part 1: Getting Started'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 1. Getting Started

## The command you just ran

먼저 처음 도커를 실행할 때 입력했던 명령여가 있습니다.
```vim
$ docker run -d -p 80:80 docker/getting-started
```
여기에 몇 가지 flag가 사용된 것을 알 수 있습니다.
- run: 컨테이너 실행
- -d : detached 모드(즉, 백그라운드)에서 컨테이너를 실행
- -p 80:80 : 호스트 포트 80을 컨테이너 포트 80에 매핑
- docker/getting-started : 사용할 이미지(해당 이미지가 설치되어 있지 않다면, 자동으로 설치)

> Tips) 단일 문자 flag를 결합하여 사용할 수 있습니다.
```vim
$ docker run -dp 80:80 docker/getting-started
```

## Docker Container Check
Docker Image를 다운로드 받고, 실행까지 완료 했습니다. 일단 백그라운드에서 실행을 했으니, Image가 설치 되었는지 확인해보겠습니다.
```vim
$ docker image ls
```
![img](/assets/img/docker/docker_image_ls.png)

위 명령어로 지금 설치되어 있는 **Docker Image**의 리스트를 확인 할 수 있습니다.


## Docker Container Check

이제 실행한 컨테이너를 확인해보겠습니다.
```vim
$ docker container ls
```
(`docker container ls`와 `docker ps`는 같은 동작을 합니다.)
![img](/assets/img/docker/docker_ps.png)

- `CONTAINER ID`: 컨테이너 별로 주어진 고유 ID
- `IMAGE`: 해당 컨테이너에서 실행 되고 있는 Docker Image
- `COMMAND`: 컨테이너가 실행될 때 컨테이너 내부에서 실행되는 명령어
- `CREATED`: 생성 된 시간
- `STATUS`: 현재 컨테이너의 상태
- `PORTS`: 현재 컨테이너의 어느 포트가, 어느 호스트 포트에 Mapping되어 있는지 나타냅니다.
- `NAMES`: 컨테이너의 이름을 나타냅니다. (실행시 지정 해 주지 않으면 자동 생성)

이제 컨테이너 내부를 들여다보겠습니다.
```vim
$ docker logs [CONTAINER_NAME | CONTAINER_ID]
```
![img](/assets/img/docker/docker_logs.png)

이 컨테이너는 Docker 홈페이지의 Getting Started 도큐먼트를 서버로 보여주는 역할을 합니다. 한 번 웹브라우저에 https://localhost 를 쳐보면 다음과 같은 화면이 나옵니다.
![img](/assets/img/docker/localhost.png)

## Container Stop
아래의 명령어를 실행하면 컨테이너를 종료할 수 있습니다.
```vim
$ docker stop [CONTAINER_NAME | CONTAINER_ID]
```
> 여기서 유의할 점은 **삭제**와 다른 개념이라는 것 입니다.

이제 컨테이너가 잘 정지되었는지 확인해보겠습니다. `-a` 플래그를 추가하여 행 중이 아닌 컨테이너를 확인할 수 있습니다.
```vim
$ docker ps -a
```

다시 컨테이너를 실행하고 싶을 경우 아래의 명령어를 입력해주면 됩니다.
``` vim
docker start [CONTAINER_NAME | CONTAINER_ID]
```



## The Docker Dashboard
도커 대시보드를 highlight하여 실행 중인 컨테이너를 빠르게 볼 수 있습니다.

컨테이너 로그에 빠르게 액세스할 수 있고, 컨테이너 내부에 shell을 가져올 수 있으며, 컨테이너 lifecycle(stop, move, etc.)를 쉽게 관리할 수 있습니다.

지금 대시 보드를 열면 tutorial이 실행되는 것을 볼 수 있습니다. 컨테이너 이름(thirsty_vaughan)은 임의로 생성 된 이름입니다. 따라서 여러분들 모두 다른 이름을 갖게 될 것입니다.

![img](/assets/img/docker/docker_start2.png)

## What is a container?
앞서 컨테이너와 이미지에 대해 간략하게 정리해둔적이 있습니다.
[참고해주세요.](https://geonkimdcu.github.io/devops/2020/12/30/docker-start/)
다시 정리하자면, 컨테이너는 호스트 시스템의 다른 모든 프로세스로부터 격리된 시스템의 또 다른 프로세스입니다.

이러한 분리는 Linux에서 오랫동안 사용되어 온 [커널 네임스페이스와 cgroup](https://medium.com/@saschagrunert/demystifying-containers-part-i-kernel-space-2c53d6979504)을 활용합니다.

## What is a container image?
컨테이너를 실행할 때는 분리된 파일 시스템을 사용합니다.

이 사용자 지정 파일 시스템은 컨테이너 이미지를 통해 제공됩니다. 이미지는 컨테이너의 파일 시스템을 포함하므로, 응용프로그램을 실행하는 데 필요한 모든 항목(모든 종속성, 구성, 스크립트, 이진 등)을 포함해야 합니다. 

또한 이미지에는 환경 변수, 실행할 기본 명령 및 기타 메타데이터와 같은 컨테이너에 대한 다른 구성도 포함됩니다.


### Layered Image

Docker가 사용하는 Disk Image는 한 장의 그림이 아니라 계층화된 이미지이며, 최종적으로 사용될 이미지는 이렇게 계층화된 이미지들의 투영으로 만들어집니다. 
 
그림으로 표현해보자면 아래와 같습니다.
기반 이미지를 두고, 그 위에 변경이나 추가, 삭제를 담은 계층들이 놓이게 되면, 그것을 위해서 투영해볼 때에는 그냥 모든 변경이 적용된 한 장의 이미지로 보이는 것입니다.

> Docker Image는 단계적으로 형성되어지는 Layered Image 입니다.

![img](/assets/img/docker/docker-layered-image.png)



## Reference
1. https://justkode.kr/cloud-computing/docker-2
2. https://www.sauru.so/blog/docker-installation-and-test-drive/