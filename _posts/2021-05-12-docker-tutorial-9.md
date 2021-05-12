---
layout: post
title: '[Docker] Part 9: Image Building Best Practices'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 9. Image Building Best Practices

## Security Scanning
이미지를 빌드한 경우 `docker scan` 명령을 사용하여 보안 취약점을 스캔하는 것이 좋습니다.

예를 들어 getting-started자습서의 앞부분에서 만든 이미지를 스캔하려면 다음을 입력하면됩니다.

```vim
$ docker scan getting-started
```


## Image Layering
이미지를 구성하는 것을 볼 수 있다는 것을 알고 계셨습니까? 은 Using `docker image history` 명령을 실행하면 이미지를 구성하는 것(이미지 내의 각 레이러르 만드는데 사용된 명령)을 볼 수 있습니다.

- `docker image history` 명령을 사용하여 getting-started자습서의 앞부분에서 만든 이미지의 레이어를 확인합니다.
```vim
$ docker image history getting-started
```

<img src="/assets/img/docker/image_history.png">
각 줄이 이미지의 레이어를 나타냅니다.


## Layer Caching
이제 레이어링이 작동하는 것을 보았으므로, 컨테이너 이미지의 빌드 시간을 줄이는 데 도움이 되는 중요한 것을 배워보겠습니다.

> 레이어가 변경되면 모든 다운 스트림 레이어도 다시 만들어야합니다.

앞서 사용했던 Dockerfile 입니다.
```vim
FROM node:12-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```

이미지의 history를 출력하기 위해 돌아가면, Dockerfile의 각 명령이 이미지의 새 layer가 되는 것을 볼 수 있습니다. 이미지를 변경했을 때, yarn 종속성을 재설치해야 했습니다. 우리가 빌드 할 때마다 동일한 종속성을 제공하는 것은 의미가 없습니다.

이 문제를 해결하기 위해선, 종속성 caching을 지원하도록 Dockerfile을 재구성해야 합니다.

```vim
FROM node:12-alpine
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --production
COPY . .
CMD ["node", "src/index.js"]
```

기존의 Dockerfile에서 `COPY package.json yarn.lock ./`이 추가되었습니다. 즉 새롭게 추가된 layer의 이전의 layer는 caching이 되고, 새롭게 추가된 layer와 그 이후의 layer만 다시 설치를 해줍니다.

```vim
Sending build context to Docker daemon  219.1kB
Step 1/6 : FROM node:12-alpine
---> b0dc3a5e5e9e
Step 2/6 : WORKDIR /app
---> Using cache
---> 9577ae713121
Step 3/6 : COPY package.json yarn.lock ./
---> bd5306f49fc8
Step 4/6 : RUN yarn install --production
---> Running in d53a06c9e4c2
yarn install v1.17.3
[1/4] Resolving packages...
[2/4] Fetching packages...
info fsevents@1.2.9: The platform "linux" is incompatible with this module.
info "fsevents@1.2.9" is an optional dependency and failed compatibility check. Excluding it from installation.
[3/4] Linking dependencies...
[4/4] Building fresh packages...
Done in 10.89s.
Removing intermediate container d53a06c9e4c2
---> 4e68fbc2d704
Step 5/6 : COPY . .
---> a239a11f68d8
Step 6/6 : CMD ["node", "src/index.js"]
---> Running in 49999f68df8f
Removing intermediate container 49999f68df8f
---> e709c03bc597
Successfully built e709c03bc597
Successfully tagged getting-started:latest
```
출력 결과를 보면 `WORKDIR /app`은 **Using cache**를 사용하고, 그 뒤의 layer는 모두 설치가 이루어지는 것을 볼 수 있습니다. 

`Using cache`를 통해 빌드 속도를 더 빠르게 할 수 있습니다. <br>
일반적으로 자주 Update 되는 layer는 뒤로, 고정적인 layer는 앞 쪽에 배치하는 것이 효율적입니다.

## Multi-Stage Builds¶
Multi-Stage 빌드는 여러 단계를 사용하여 이미지를 만드는 데 도움이되는 매우 강력한 도구입니다. 아래와 같이 몇 가지 장점이 있습니다.

- 런타임 종속성에서 빌드 시간 종속성 분리
- app 실행에 오직 필요한 것만 전달해줌으로써 전반적인 이미지 사이즈를 줄일 수 있습니다.

### Maven/Tomcat Example

```vim
FROM maven AS build
WORKDIR /app
COPY . .
RUN mvn package

FROM tomcat
COPY --from=build /app/target/file.war /usr/local/tomcat/webapps 
```

위 소스 코드를 보면 두 단계로 이루어져서 이미지가 경량화 되었음을 알 수 있습니다.

maven을 먼저 빌드하여 tomcat 단계에서 그 결과를 copy하여 `/usr/local/tomcat/webapps`로 저장한다는 내용입니다.


## Recap
이미지의 구조에 대해 조금 이해하면 이미지를 더 빠르게 구축하고 변경 사항을 줄일 수 있습니다. 

이미지를 스캔하면 실행 및 배포하는 컨테이너가 안전하다는 확신을 갖게됩니다. 

multi-stage 빌드 또한 런타임 종속성에서 빌드 시간 종속성을 분리하여 전체 이미지 크기를 줄이고 최종 컨테이너 보안을 높이는 데 도움이됩니다.

<br><br>

## Reference
1. http://localhost/tutorial/image-building-best-practices/

