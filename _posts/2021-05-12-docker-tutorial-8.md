---
layout: post
title: '[Docker] Part 08: Using Docker Compose'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 8. Using Docker Compose
**Docker Compose** 는 다중 컨테이너 애플리케이션을 정의하고 공유 할 수 있도록 개발 된 도구입니다. Compose를 사용하면 YAML 파일을 생성하여 서비스를 정의하고 단일 명령으로 모든 것을 스핀 업하거나 모두 분해 할 수 있습니다.

Compose 사용의 가장 큰 장점은 파일에 애플리케이션 스택을 정의하고 이를 프로젝트 저장소의 루트에 보관하고 (이제 버전 관리 됨) 다른 사람이 프로젝트에 쉽게 기여할 수 있도록 할 수 있다는 것입니다. <br>
누군가는 저장소를 복제하고 작성된 앱을 시작하기만 하면 됩니다.

## Installing Docker Compose
Windows 또는 Mac 용 Docker Desktop / Toolbox를 설치했다면 이미 Docker Compose가 있습니다! Play-with-Docker 인스턴스에는 이미 Docker Compose도 설치되어 있습니다. 

버전 정보를 볼 수 있습니다.
```vim
$ docker-compose version
```
![img](/assets/img/docker/compose.png)

## Creating our Compose File
1. app project의 루트에서, `docker-compose.yml`이란 이름으로 파일을 생성합니다.

2. 작성 파일에서 먼저 스키마 버전을 정의합니다.

```yml
version: "3.7"
```

3. 다음으로 애플리케이션의 일부로 실행하려는 서비스 (또는 컨테이너) 목록을 정의합니다.

```yml
version: "3.7"

services:
```

이제 한 번에 서비스를 작성 파일로 마이그레이션하기 시작합니다.

## Defining the App Service

아래의 명령어는 앱 컨테이너를 정의하는데 사용했던 것입니다.

```vim
$docker run -dp 3000:3000 \
  -w /app -v "$(pwd):/app" \
  --network todo-app \
  -e MYSQL_HOST=mysql \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=secret \
  -e MYSQL_DB=todos \
  node:12-alpine \
  sh -c "yarn install && yarn run dev"
```

1. 먼저 컨테이너의 서비스 항목과 이미지를 정의하겠습니다. 우리는 서비스의 이름을 선택할 수 있습니다. 이름은 자동으로 네트워크 별칭이되어 MySQL 서비스를 정의 할 때 유용합니다.

```yml
version: "3.7"

services:
  app:
    image: node:12-alpine
```

2. 일반적으로 image 주문에 대한 요구 사항은 없지만 정의에 가까운 명령이 표시됩니다.

```yml
version: "3.7"

services:
  app:
    image: node:12-alpine
    command: sh -c "yarn install && yarn run dev"
```

3. 서비스에 대한 포트를 정의하여 `-p 3000:3000` 일부를 마이그레이션 하겠습니다.

```yml
version: "3.7"

services:
  app:
    image: node:12-alpine
    command: sh -c "yarn install && yarn run dev"
    ports:
      - 3000:3000
```

4. 다음으로 `working_dir`와 `volumnes` 정의를 사용함으로써 working directory(`w /app`)와 volume mapping(`-v "$(pwd):/app"`)을 마이그레이션 합니다.

Docker Compose 볼륨 정의의 한 가지 장점은 현재 디렉터리의 상대 경로를 사용할 수 있다는 것입니다.

```yml
version: "3.7"

services:
  app:
    image: node:12-alpine
    command: sh -c "yarn install && yarn run dev"
    ports:
      - 3000:3000
    working_dir: /app
    volumes:
      - ./:/app
```

5. 마지막으로 `environment`키를 사용하여 환경 변수 정의를 마이그레이션 해야합니다.

```yml
version: "3.7"

services:
  app:
    image: node:12-alpine
    command: sh -c "yarn install && yarn run dev"
    ports:
      - 3000:3000
    working_dir: /app
    volumes:
      - ./:/app
    environment:
      MYSQL_HOST: mysql
      MYSQL_USER: root
      MYSQL_PASSWORD: secret
      MYSQL_DB: todos
```

## Defining the MySQL Service
이제 MySQL Service를 정의해보겠습니다. 

해당 컨테이너에 사용한 명령어는 다음과 같습니다.

```vim
$ docker run -d \
  --network todo-app --network-alias mysql \
  -v todo-mysql-data:/var/lib/mysql \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=todos \
  mysql:5.7
```

1. 먼저 새 서비스를 정의하고 이름을 `mysql`로 지정하여 네트워크 별칭을 자동으로 가져옵니다. 계속해서 사용할 이미지도 지정하겠습니다.

```yml
version: "3.7"

services:
  app:
    # The app service definition
  mysql:
    image: mysql:5.7
```

2. 다음으로 volume mapping을 정의합니다. `docker run`으로 컨테이너를 실행했을 때 named volume이 자동으로 생성되었습니다. <br>
그러나 Compose로 실행할 때는 발생하지 않습니다. `volumes:`라는 최상위 섹션에서 볼륨을 정의한 다음 서비스 구성에서 mountpoint를 지정해야합니다. 단순히 볼륨 이름 만 제공하면 기본 옵션이 사용됩니다. 여기에 더 [많은 옵션](https://docs.docker.com/compose/compose-file/#volume-configuration-reference)을 알아볼 수 있습니다.

```yml
version: "3.7"

services:
  app:
    # The app service definition
  mysql:
    image: mysql:5.7
    volumes:
      - todo-mysql-data:/var/lib/mysql

volumes:
  todo-mysql-data:
```


3. 마지막으로 환경 변수만 지정하면됩니다.

```yml
version: "3.7"

services:
  app:
    # The app service definition
  mysql:
    image: mysql:5.7
    volumes:
      - todo-mysql-data:/var/lib/mysql
    environment: 
      MYSQL_ROOT_PASSWORD: secret
      MYSQL_DATABASE: todos

volumes:
  todo-mysql-data:
```

최종 완성된 `docker-compose.yml` 모습은 다음과 같아야합니다.

```yml
version: "3.7"

services:
  app:
    image: node:12-alpine
    command: sh -c "yarn install && yarn run dev"
    ports:
      - 3000:3000
    working_dir: /app
    volumes:
      - ./:/app
    environment:
      MYSQL_HOST: mysql
      MYSQL_USER: root
      MYSQL_PASSWORD: secret
      MYSQL_DB: todos

  mysql:
    image: mysql:5.7
    volumes:
      - todo-mysql-data:/var/lib/mysql
    environment: 
      MYSQL_ROOT_PASSWORD: secret
      MYSQL_DATABASE: todos

volumes:
  todo-mysql-data:
```

## Running our Application Stack
이제 `docker-compose.yml` 파일이 있으니 시작해보겠습니다.

1. 먼저 app 또는 db의 다른 복사본이 실행되고 있지 않는지 확인해봅니다.

2. `docker-compose up` 명령어를 통해 애플리케이션 스택을 시작합니다. 거기에 백그라운드에서 모든 것을 실행하도록 `-d` flag를 추가합니다.

```vim
$ docker-compose up -d
```
이것을 실행하면 다음과 같은 화면이 출력됩니다.
```vim
Creating network "app_default" with the default driver
Creating volume "app_todo-mysql-data" with default driver
Creating app_app_1   ... done
Creating app_mysql_1 ... done
```

즉 볼륨과 네트워크가 생성된 것을 알 수 있습니다. 기본적으로 `Docker Compose`는 애플리케이션 스택을 위해 특별히 네트워크를 자동으로 생성합니다 (이것이 compose 파일에서 정의하지 않은 이유입니다).

3. `docker-compose logs -f`명령을 사용하여 로그를 살펴 보겠습니다. 단일 스트림으로 인터리브된 각 서비스의 로그를 볼 수 있습니다. 

4. 이제 앱을 열고 실행중인 것을 볼 수 있어야 합니다.

## Seeing our App Stack in Docker Dashboard
docker Dashboard를 보면 app 이라는 그룹이 있음을 알 수 있습니다. Docker Compose의 "프로젝트 이름"이며 컨테이너를 함께 그룹화하는 데 사용됩니다. 기본적으로 프로젝트 이름은 `docker-compose.yml`이(가)있는 디렉터리의 이름입니다.

앱을 아래로 돌리면 compose 파일에서 정의한 두 개의 컨테이너가 표시됩니다.

이름은 `<project-name>_<service-name>_<replica-number>`패턴을 따르기 때문에 좀 더 설명적입니다. 따라서 어떤 컨테이너가 우리 앱이고 어떤 컨테이너가 mysql 데이터베이스인지 매우 쉽고 빠르게 확인할 수 있습니다.

<img src="/assets/img/docker/capture05.png">

## Tearing it All Down
모든 것을 해체할 준비가 되었다먼, `docker compose down`을 실행하거나 대시보드 휴지통을 누르면 됩니다. 컨테이너가 중지되고 네트워크가 제거될 것 입니다.

> Tips) 기본적으로 compose 파일의 named volume은 `docker-compose down` 을 실행할 때 제거되지 않습니다. 볼륨을 제거하려면 --volumes 플래그를 추가해야합니다.

Docker Dashboard는 앱 스택을 삭제할 때 볼륨을 제거 하지 않습니다.

## Recap
이번 장에서는 Docker Compose에 대해 알아보고, 이를 통해 multi service 애플리케이션의 정의 및 공유를 대폭 단순화하는 방법을 배웠습니다. 사용하던 명령을 적절한 compose 형식으로 변환하여 Compose 파일을 만들었습니다.



<br><br>

## Reference
1. http://localhost/tutorial/using-docker-compose/