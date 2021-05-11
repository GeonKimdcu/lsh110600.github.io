---
layout: post
title: '[Docker] Part 2: Our Application'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 2. Our Application

Node.js에서 실행되는 간단한 todo list 관리자로 작업 할 것입니다.

![img](/assets/img/docker/todo-list-sample.png)

## Getting our App
application을 실행하려면 먼저 컴퓨터에 application source code를 가져와야합니다. 실제 프로젝트의 경우 일반적으로 저장소를 복제합니다. 

이 튜토리얼에는 application이 포함된 ZIP파일이 있습니다.

1. [ZIP을 다운로드하십시오](http://localhost/assets/app.zip).
    - 터미널에서 `wget`을 사용해 다운로드해줍니다.
    ![img](/assets/img/docker/unzip.png)

    - `unzip`을 사용해 압축을 풀어줍니다.
    ![img](/assets/img/docker/wget.png)

2. 편집기가 필요한 경우 Visual Studio Code를 사용할 수 있습니다. `package.json`및 두 개의 하위 디렉터리 (`src`및 `spec`) 가 표시되어야합니다.
![img](/assets/img/docker/ide-screenshot.png)

<br>

## Building the App's Container Image

application을 빌드하려면 `Dockerfile`이 필요합니다. Dockerfile은 컨테이너 이미지를 만드는 데 사용되는 지침의 텍스트 기반 스크립트입니다. 

1. ```Dockerfile```이란 파일명으로  ```package.json```과 동일한 경로에 vim을 사용해 만들어줍니다.
```vim
$ vim Dockerfile
```
아래의 코드를 작성해줍니다.
```vim
FROM node:12-alpine
RUN apk add --no-cache python g++ make
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```
> 만약 다음 단계에 에러가 발생할 경우 `.txt` 확장자를 가진 `Dockerfile`이 있는지 확인해보세요.

2. 이제 ```docker build``` 명령을 사용하여 컨테이너 이미지를 빌드하겠습니다.

```vim
$ docker build -t getting-started .
```

위 명령어를 통해 Dockerfile을 사용하여 새 컨테이너 이미지를 빌드해줍니다.

많은 "Layers"가 다운로드됩니다. 이는 빌더에게 ```node:12-alpine```이미지에서 시작하고 싶다고 지시했기 때문입니다. 

그러나 우리 컴퓨터에는 이 파일이 없었기 때문에 해당 이미지를 다운로드해야했습니다.

image를 다운로드 한 후 application을 복사하고 애플리케이션의 종속성을 설치하기 위해 `yarn`을 사용했습니다. `CMD`는 이미지에서 컨테이너를 시작할 때 실행하기 위한 기본 명령어를 지정해줍니다.

마지막으로 `-t`는 이미지에 태그를 지정합니다. 이것은 최종 이미지에 대해 사람이 읽을 수 있는 이름으로 생각하면됩니다.

이미지 이름을 `getting-started`로 지정 했으므로 컨테이너를 실행할 때 해당 이미지를 참조 할 수 있습니다.

마지막 `.`은 현재 디렉토리에 `Dockerfile`을 찾는 것 입니다.

## Starting an App Container
이제 이미지가 있으니 application을 실행해보겠습니다.
이를 위해 `docker run` 명령어를 사용해줍니다.

1. `docker run` 명령을 사용하여 컨테이너를 시작 하고 방금 만든 이미지의 이름을 지정합니다.
```vim
$ docker run -dp 3000:3000 getting-started
```
- `-d` 및 `-p` flags를 통해 새 컨테이너를 **detached mode**(백그라운드)로 실행하고 호스트의 포트 3000과 컨테이너의 포트 3000를 매핑해줍니다. 포트 매핑이 없으면 application에 액세스 할 수 없습니다.

2. http://localhost:3000 으로 이동 합니다. 우리 앱이 보여야합니다!
![img](/assets/img/docker/todo-list-empty.png)

3. 항목을 여러 개 추가하고 잘 작동하는지 확인해줍니다. 우리가 입력한 프런트 엔드가 백엔드에 항목을 성공적으로 저장하고 있는 것을 볼 수 있습니다.

이제 Docker Dashboard를 보면, 두 개의 컨테이너가 실행중인 것을 볼 수 있습니다.(tutorial, Todo app)
![img](/assets/img/docker/dashboard.png)

## Recap
이번 파트에서는 컨테이너 이미지 빌드에 대한 기본 사항을 배웠고 이를 위해 Dockerfile을 만들었습니다.

이미지를 빌드 한 후 컨테이너를 시작하고 실행중인 앱을 확인했습니다!

다음으로 앱을 수정하고 실행중인 애플리케이션을 새 이미지로 업데이트하는 방법을 알아보겠습니다. 그 과정에서 몇 가지 다른 유용한 명령을 배우게됩니다.

## Reference
1. http://localhost/tutorial/our-application/#starting-an-app-container
