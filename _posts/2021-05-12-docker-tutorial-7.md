---
layout: post
title: '[Docker] Part 7: Multi-Container Apps'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 7. Multi-Container Apps
지금까지 우리는 단일 컨테이너 app을 작업해왔습니다. 그러나 이제 application stack에 MySQL을 추가하려고 합니다. <br>
여기서 이제 동일한 컨테이너에 실행할지 혹은 별도의 컨테이너에 MySQL을 실행할지 생각해보아야 합니다.

일반적으로 각 컨테이너는 한 가지 일을 잘 수행해야 합니다. 그 이유는 아래와 같습니다.
- 데이터베이스와는 다르게 API 및 프런트 엔드를 확장해야 할 좋은 기회가 있습니다.
- 별도의 컨테이너를 사용하여 버전을 격리하고 업데이트 할 수 있습니다.
- 여러 프로세스를 실행하려면 프로세스 관리자(컨테이너가 하나의 프로세스 만 시작)가 필요하므로 컨테이너 시작 / 종료가 복잡해집니다.

아래 그림과 같이 application을 업데이트 해보겠습니다.
![img](/assets/img/docker/multi-app-architecture.png)

## Container Networking
기본적으로 컨테이너는 격리 된 상태로 실행되며 동일한 머신의 다른 프로세스, 컨테이너에 대해 알지 못합니다. 그렇다면 한 컨테이너가 다른 컨테이너와 어떻게 통신할 수 있을까요? 대답은 **네트워킹** 입니다.

> 두 개의 컨테이너가 동일한 네트워크에 있는 경우, 서로 통신 할 수 있습니다. 그렇지 않으면 할 수 없습니다.

## Starting MySQL
네트워크에 컨테이너를 배치하는 방법에는 1) 시작할 때 할당하거나 2) 기존 컨테이너를 연결하는 두 가지 방법이 있습니다. <br>
지금은 네트워크를 먼저 만들고 시작할 때 MySQL 컨테이너를 연결합니다.

1. 먼저 네트워크를 생성합니다.
```vim
$ docker network create todo-app
```

2. MySQL 컨테이너를 시작하고 네트워크를 연결해줍니다.
```vim
$ docker run -d \
    --network todo-app --network-alias mysql \
    -v todo-mysql-data:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=secret \
    -e MYSQL_DATABASE=todos \
    mysql:5.7
```
> Tips) `todo-mysql-data` named volumed을 사용하고, MySQL의 데이터를 저장하는 곳인 `/var/lib/mysql`에 마운트합니다. 
<br> 여기서 우리는 `docker volume create` 명령을 실행하지 않았습니다. <br> Docker는 우리가 named volume을 사용하고자 한다는 것을 인식하고 자동으로 생성합니다.

3. 데이터베이스에 연결한 뒤, 잘 연결되었는지 확인해줍니다.
```vim
$ docker exec -it <mysql-container-id> mysql -p
```
암호 프롬프트가 나타나면 **secret**을 입력하십시오. MySQL 셸에서 데이터베이스를 나열하고 `todo` 데이터베이스가 표시되는지 확인합니다.

```sql
mysql> SHOW DATABASES;
```

다음과 같은 창이 출력되는지 확인합니다.
```vim
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
| todos              |
+--------------------+
5 rows in set (0.00 sec)
```

## Connecting to MySQL
이제 MySQL이 실행되고 있다는 것을 알았으니 사용해 보겠습니다.

동일한 네트워크에서 다른 컨테이너를 실행하는 경우 어떻게 컨테이너를 찾을 수 있을까요? 각 컨테이너에는 고유한 IP 주소가 있음을 기억하세요.

이를 파악하기 위해 네트워킹 문제를 해결하거나 디버깅하는데 유용한 많은 도구와 함께 제공되는 [nicolaka / netshoot](https://github.com/nicolaka/netshoot) 컨테이너를 사용할 것입니다 .

1. nicolaka / netshoot 이미지를 사용하여 새 컨테이너를 시작합니다. 동일한 네트워크에 연결해야합니다.
```vim
$ docker run -it --network todo-app nicolaka/netshoot
```

2. 컨테이너 내부에서 유용한 DNS 도구인 `dig` 명령을 사용할 것 입니다. 그리고 `mysql` 호스트 이름에 대한 IP 주소를 조회합니다.
```vim
$ dig mysql
```
그러면 아래와 같이 출력될 것 입니다.
```vim
; <<>> DiG 9.14.1 <<>> mysql
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 32162
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;mysql.             IN  A

;; ANSWER SECTION:
mysql.          600 IN  A   172.23.0.2

;; Query time: 0 msec
;; SERVER: 127.0.0.11#53(127.0.0.11)
;; WHEN: Tue Oct 01 23:47:24 UTC 2019
;; MSG SIZE  rcvd: 44
```
`ANSWER SECTION` 에서 `172.23.0.2` IP address로 보이는 mysql의 `A` 레코드를 확인할 수 있습니다. <br>
mysql은 일반적으로 유효한 호스트 네임이 없지만, 도커는 network alias를 가졌으며 container의 IP address를 확인할 수 있습니다. 

즉, application은 단순히 `mysql`이란 이름의 호스트와 연결하면 데이터베이스와 통신합니다.

## Running our App with MySQL
To do app은 MySQL 연결 설정을 지정하기 위해 몇 가지 환경 변수 설정을 지원합니다.
- `MYSQL_HOST`: 실행중인 MySQL 서버의 호스트 이름
- `MYSQL_USER`: 연결에 사용할 사용자 이름
- `MYSQL_PASSWORD`: 연결에 사용할 비밀번호
- `MYSQL_DB`: 연결하여 사용할 데이터베이스

이제 dev-ready 컨테이너를 시작해보겠습니다.

1. 위의 각 환경 변수를 지정하고 컨테이너를 앱 네트워크에 연결합니다.
```vim
$ docker run -dp 3000:3000 \
  -w /app -v "$(pwd):/app" \
  --network todo-app \
  -e MYSQL_HOST=mysql \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=secret \
  -e MYSQL_DB=todos \
  node:12-alpine \
  sh -c "yarn install && yarn run dev"
```
2. 컨테이너(`docker logs <container-id>`)에 대한 로그를 보면 mysql 데이터베이스를 사용하고 있음을 나타내는 메시지가 표시됩니다.
```vim
# Previous log messages omitted
$ nodemon src/index.js
[nodemon] 1.19.2
[nodemon] to restart at any time, enter `rs`
[nodemon] watching dir(s): *.*
[nodemon] starting `node src/index.js`
Connected to mysql db at host mysql
Listening on port 3000
```

3. 브라우저에서 앱을 열고 to do list에 몇 가지 항목을 추가합니다.

4. mysql 데이터베이스에 연결하고 항목이 데이터베이스에 기록되고 있음을 증명합니다. 비밀번호는 secret 입니다.
```vim
$ docker exec -it <mysql-container-id> mysql -p todos
```

그리고 mysql 셸에서 다음을 실행합니다.
```sql
mysql> select * from todo_items;
```
```vim
+--------------------------------------+--------------------+-----------+
| id                                   | name               | completed |
+--------------------------------------+--------------------+-----------+
| c906ff08-60e6-44e6-8f49-ed56a0853e85 | Do amazing things! |         0 |
| 2912a79e-8486-4bc3-a4c5-460793a575ab | Be awesome!        |         0 |
+--------------------------------------+--------------------+-----------+
```

Docker Dashboard를 살펴보면 두 개의 앱 컨테이너가 실행되고 있음을 알 수 있습니다. 그러나 단일 앱에서 함께 그룹화된다는 실제 징후는 없습니다. 이를 개선하는 방법을 곧 살펴 보겠습니다!

![img](/assets/img/docker/dashboard-multi-container-app.png)

## Recap
이제 별도의 컨테이너에서 실행되는 외부 데이터베이스에 데이터를 저장하는 애플리케이션이 있습니다. 컨테이너 네트워킹에 대해 배웠고 DNS를 사용하여 서비스 검색을 수행하는 방법을 살펴 보았습니다.

그러나 응용 프로그램을 시작하기 위해서 네트워크를 생성하고, 컨테이너를 시작하고, 모든 환경 변수를 지정하고, 포트를 노출하는 등의 작업을해야합니다! 기억해야 할 것이 많고 다른 사람에게 전달하기가 더 어렵습니다.

다음 섹션에서는 **Docker Compose**에 대해 설명합니다. Docker Compose를 사용하면 애플리케이션 스택을 훨씬 더 쉽게 공유 할 수 있으며 다른 사람들이 단일 (및 간단한) 명령으로 스택을 가동 할 수 있습니다!



<br><br>
## Reference
1. http://localhost/tutorial/multi-container-apps/
