---
layout: post
title: '[Docker] Mac OS에서 도커 설치 후 실행'
subtitle: 'Mac OS에서 도커 설치 후 실행'
categories: devops
tags: doker
comments: true
---

도커의 기본적인 사용법에 대해 알아봅니다.

## Mac os에서 도커 설치하기
도커 정식버전에는 Apple Silicon(M1 칩) 버전이 없지만 Preview 버전이 Apple Silicon 버전을 지원하기때문에 [도커 사이트](https://docs.docker.com/docker-for-mac/apple-silicon/)에 들어가서 M1 Tech Preview 버전을 받아서 설치합니다.

다운 받은 dmg파일을 클릭하여 도커를 설치해줍니다.

![img](/assets/img/docker/docker_download.png)

도커를 설치하고, 클릭해보면 아래같은 화면이 보입니다.<br>
터미널에서 아래 화면에 뜬 명령어를 실행하고 localhost로 접속하면 실행이 잘되는 것을 확인 할 수 있습니다.

`$ docker run -it --rm -p 80:80 docker/getting-started`

![img](/assets/img/docker/docker_start.png)

![img](/assets/img/docker/docker_start2.png)

> Local host 접속화면

![img](/assets/img/docker/localhost.png)

다음 시간에는 도커 공식 도큐먼트 [Getting Started](https://docs.docker.com/get-started/)를 한 단계씩 배워보겠습니다.


<br><br>
## Reference
1. https://itkoo.tistory.com/10