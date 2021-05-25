---
layout: post
title: '[Docker] Part 03: Updating our App'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 3. Updating our App

## Updating our Source Code

1. `src/static/js/app.js`에서 아래와 같이 수정합니다.
```html
- <p className="text-center">No items yet! Add one above!</p>
+ <p className="text-center">You have no todo items yet! Add one above!</p>
```

2. 이미지의 업데이트 된 버전을 빌드 해 보겠습니다.
```vim
$ docker build -t getting-started .
```

3. 업데이트 된 코드를 사용하여 새 컨테이너를 시작하겠습니다.
```vim
$ docker run -dp 3000:3000 getting-started
```

> 세 번째 명령어를 입력하면 아래와 같은 에러를 볼 수 있습니다.
>   > 기존 컨테이너가 아직 가동 중이기 때문에 새 컨테이너를 시작할 수 없다는 에러입니다. <br>
이유는 컨테이너가 호스트의 포트 3000을 사용하고 있고 기계에서 하나의 프로세스(컨테이너 포함)만 특정 포트를 취할 수 있기 때문입니다. 이 문제를 해결하려면 기존 컨테이너를 제거해야 합니다.

![img](/assets/img/docker/error_1.png)

## Replacing our Old Container
컨테이너를 제거하기 위해선 먼저 컨테이너를 중지해야 합니다.

컨테이너를 제거할 수 있는 방법에는 2가지가 있습니다.

### 1. Removing a container using the CLI
1. `docker ps` 명령을 사용하여 컨테이너의 ID를 가져옵니다.
```vim
$ docker ps
```

2. `docker stop` 명령을 사용하여 컨테이너를 중지합니다.
```vim
# Swap out <the-container-id> with the ID from docker ps
$ docker stop <the-container-id>
```

3. 컨테이너가 중지되면 `docker rm` 명령을 사용하여 제거 할 수 있습니다.
```vim
$ docker rm <the-container-id>
```

> Tips) 명령에 "**force**" flag를 추가하여 단일 명령으로 컨테이너를 중지하고 제거 할 수 있습니다. `docker rm -f <the-container-id>`


### 2. Removing a container using the Docker Dashboard
Docker Dashboard를 사용하면 컨테이너의 ID를 찾아서 제거하는 것보다 훨씬 쉽게 제거할 수 있습니다.

![img](/assets/img/docker/remove_container.png)

### 3. Starting our updated app container
1. 이제 업데이트된 App을 실행해보겠습니다.

```vim
docker run -dp 3000:3000 getting-started
```

2. http://localhost:3000 에서 브라우저를 새로 고치면 업데이트 된 도움말 텍스트가 표시됩니다.
![img](/assets/img/docker/todo-list-updated-empty-text.png)

업데이트 후 To do list에서 항목들이 모두 사라진 것을 알 수 있습니다.

## Recap
다음 섹션에서는 변경할 때마다 새 컨테이너를 다시 빌드하고 시작할 필요없이 코드 업데이트를 보는 방법에 대해 알아보겠습니다.


<br><br>

## Reference
1. http://localhost/tutorial/updating-our-app/#updating-our-source-code
