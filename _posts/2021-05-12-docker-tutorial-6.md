---
layout: post
title: '[Docker] Part 06: Using Bind Mounts'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 6. Using Bind Mounts
이전 장에서는 데이터베이스에 데이터를 유지 하기 위해 named volume을 사용했습니다. 데이터가 저장 되는 위치에 대해 걱정할 필요가 없기 때문에 단순히 데이터를 저장하려는 경우 named volume이 좋습니다.

**bind mounts**를 사용 하면 호스트의 정확한 마운트 지점을 제어합니다.<br>
이를 사용하여 데이터를 유지할 수 있지만 컨테이너에 추가 데이터를 제공할 때도 종종 사용할 수 있습니다. 
<br>
애플리케이션에서 작업 할 때 bind mounts를 사용하여 소스 코드를 컨테이너에 마운트하여 코드 변경 사항을 보고 응답하고 변경 사항을 즉시 확인할 수 있습니다.

노드 기반 응용 프로그램의 경우 [nodemon](https://www.npmjs.com/package/nodemon)은 파일의 수정 사항을 감시 한 다음 응용 프로그램을 다시 시작하는 데 유용한 도구입니다.

## Quick Volume Type Comparisons
bind mounts 및 named volume은 Docker 엔진과 함께 제공되는 두 가지 주요 볼륨 유형입니다.

![img](/assets/img/docker/volume.png)

## Starting a Dev-Mode Container
개발 workflow를 지원하기 위해 컨테이너를 실행하려면 아래와 같이 수행해야 합니다.
- 컨테이너에 소스 코드를 mount 해야합니다.
- **"dev"** 종속성을 포함하여, 모든 종속성을 설치해야합니다.
- 파일시스템 변화를 감지하기 위해 **nodemon**을 시작해야합니다.

이제 진행해보겠습니다.

1. `getting-started` 가 실행중인 이전 컨테이너가 없는지 확인해줍니다.

2. 아래의 명령어를 실행해줍니다.
```vim
$ docker run -dp 3000:3000 \
    -w /app -v "$(pwd):/app" \
    node:12-alpine \
    sh -c "yarn install && yarn run dev"
```

- `-dp 3000:3000`: detached mode에서 실행하고 포트 매핑 생성
- `-w /app`: **working directory** 또는 명령이 실행될 현재 디렉토리를 설정합니다.
- `-v "$(pwd):/app"`: 현재 디렉토리를 `/app`으로 마운트해줍니다.
- `node:12-alpine`: Dockerfile의 앱에 대한 기본 이미지로 이 이미지를 사용해줍니다.
- `sh -c "yarn install && yarn run dev"`: 우선 `sh`사용해 shell을 실행합니다. 모든 종속성을 설치하기 위해서 `yarn install`을 실행하고 그런 다음 `yarn run dev`을 실행해줍니다. 

3. `docker logs -f <container-id>`을 사용해 로그를 볼 수 있습니다. 

```vim
docker logs -f <container-id>
$ nodemon src/index.js
[nodemon] 1.19.2
[nodemon] to restart at any time, enter `rs`
[nodemon] watching dir(s): *.*
[nodemon] starting `node src/index.js`
Using sqlite database at /etc/todos/todo.db
Listening on port 3000
```

4. 이제 앱을 변경해보겠습니다. `src/static/js/app.js` 파일에서, "Add Item" 버튼을 단순히 "Add"로 수정해줍니다.
```js
- {submitting ? 'Adding...' : 'Add Item'}
+ {submitting ? 'Adding...' : 'Add'}
```

5. 변경되었는지 새로고침 후 확인해줍니다.

6. 이제 컨테이너를 정지하고, `docker build -t getting-started .`명령어를 실행하여 새 이미지를 빌드해줍니다.


bind mount를 사용하는 것은 로컬 개발 설정에서 매우 일반적입니다. 

장점은 개발 시스템에 모든 빌드 도구와 환경을 설치할 필요가 없다는 것입니다. `docker run` 단일 명령으로 개발 환경을 가져와 사용할 수 있습니다. 

이후 단계에서 **Docker Compose**에 대해 배워보겠습니다. 이는 명령을 단순화하는 데 도움이 될 것입니다.

##  Recap
production을 준비하려면 데이터베이스를 SQLite에서 작업하던 것에서 좀 더 확장 할 수 있는 곳으로 migration 해야합니다. <br>
단순화를 위해 관계형 데이터베이스를 유지하고 애플리케이션을 MySQL을 사용하도록 전환합니다. 

하지만 MySQL을 어떻게 실행해야할까요? 컨테이너가 서로 통신하도록 하려면 어떻게 해야하는지 다음 장에서 이야기하겠습니다!

***
참고로 `docker ps -a`를 실행하면 이전의 컨테이너까지 모두 보여줍니다.
호스트에서 이러한 컨테이너를 기록하고 있기 때문에 모두 제거하고 싶을 경우,
`docker container prune`을 실행하면 됩니다.


<br><br>

## Reference
1. http://localhost/tutorial/using-bind-mounts/

