---
layout: post
title: '[Docker] Part 04: Sharing our App'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 4. Sharing our App
이미지를 구축했으니 이제 이를 공유해보겠습니다. 도커 이미지를 공유하기 위해선 **Docker registry**를 사용해야 합니다.

기본 레지스트리는 **Docker Hub**이며, 이곳에서 우리가 모든 이미지를 불러올 수 있습니다.

## Create a Repository
이미지를 **push** 하려면 먼저 Docker Hub에 리포지토리를 만들어야합니다.

1. [Docker Hub](https://hub.docker.com/)로 이동하여 로그인을 합니다.

2. **Create Repository** 버튼을 클릭합니다.

3. 리포지토리 이름의 경우 **getting-started**로 하고, 공개설정이 **public**으로 되어 있는지 확인한다.

4. **Create** 버튼을 클릭해줍니다.

페이지 오른쪽을 보면 **Docker commands** 라는 섹션이 표시 됩니다. 이 저장소에 **Push**하기 위해 실행해야하는 예제 명령이 제공됩니다.
![img](/assets/img/docker/push-command.png)

## Pushing our Image
1. Docker Hub에 표시되는 push 명령을 실행 해보겠습니다. 명령은 "docker"가 아닌 네임 스페이스를 사용합니다.

```vim
$ docker push docker/getting-started
The push refers to repository [docker.io/docker/getting-started]
An image does not exist locally with the tag: docker/getting-started
```
왜 실패 했을까요? push 명령이 docker/getting-started라는 이미지를 찾지 못했습니다. `docker image ls`를 실행하면 하나도 표시되지 않습니다.

문제를 해결하려면 우리가 만든 기존 이미지에 다른 이름을 부여하기 위해 "태그"를 지정해야합니다.

2. `docker login -u YOUR-USER-NAME` 명령어를 사용하여 Docker HUb에 로그인합니다.
![img](/assets/img/docker/login.png)

3. `getting-started` 이미지에 새 이름을 부여하기 위해 `docker tag` 명령어를 사용합니다. **YOUR-USER-NAME에 Docker ID로 교체해야합니다.**

4. 이제 다시 push 명령어를 실행해보겠습니다.
- Docker Hub에서 값을 복사하는 tagname 경우 이미지 이름에 태그를 추가하지 않았으므로 해당 부분을 삭제할 수 있습니다.
- 태그를 지정하지 않으면 latest 라는 태그를 사용합니다.

![img](/assets/img/docker/capture01.png)

## Running our Image on a New Instance
이제 이미지가 빌드되고 레지스트리로 푸시되었으므로 컨테이너 이미지를 새로운 인스턴스에서 앱을 실행해 보겠습니다!

1. [Play with Docker](https://labs.play-with-docker.com/)로 접속 후 로그인을 합니다.

2. 왼쪽 사이드 바에서 "+ ADD NEW INSTANCE" 버튼을 클릭하세요. 몇 초 후에 브라우저에서 터미널 창이 열립니다.

![img](/assets/img/docker/play_with_docker.png)

3. 터미널에서 새로 푸시 된 앱을 시작합니다.
```vim
$ docker run -dp 3000:3000 YOUR-USER-NAME/getting-started
```

4. 3000 badge를 클릭하면 수정된 앱이 나타날 것 입니다.
3000 badge가 표시되지 않으면 "Open Port" 버튼을 클릭하고 3000을 입력하면 됩니다.

## Recap
이번 파트에서는 이미지를 레지스트리로 푸시하여 공유하는 방법을 배웠습니다.

그런 다음 새로운 인스턴스로 이동하여 새로 푸시 된 이미지를 실행해보았습니다.

이는 파이프 라인이 이미지를 생성하고 레지스트리로 푸시 한 다음 프로덕션 환경에서 최신 버전의 이미지를 사용할 수있는 CI 파이프 라인에서 매우 일반적입니다.

이제 저번 파트의 끝에서 발견 한 내용으로 돌아가보겠습니다. 앱을 다시 시작했을 때 모든 To do list 항목이 손실되었음을 알았습니다. 다음 장에서는 다시 시작할 때 데이터를 유지할 수있는 방법을 알아 보겠습니다!


<br><br>

## Reference
1. http://localhost/tutorial/sharing-our-app/