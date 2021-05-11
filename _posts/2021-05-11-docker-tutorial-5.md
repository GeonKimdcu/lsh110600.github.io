---
layout: post
title: '[Docker] Part 5: Persisting our DB'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 5. Persisting our DB
컨테이너를 시작할 때마다 To do list가 초기화됩니다.

## The Container's Filesystem
컨테이너가 실행될 때 파일 시스템에 대한 이미지의 다양한 계층을 사용합니다.

각 컨테이너는 파일을 **create/update/remove** 하기위한 자체 **scratch space**도 갖습니다. 동일한 이미지를 사용하더라도 변경 사항은 다른 컨테이너에 표시되지 않습니다.

### Seeing this in Practice
위 사실을 확인하기 위해 두 개의 컨테이너를 시작하고 각각의 파일을 생성해보겠습니다. 한 컨테이너에서 생성된 파일을 다른 컨테이너에서 사용할 수 없다는 것을 알 수 있습니다.

1. ubuntu 컨테이너를 시작해 1에서 10000 사이의 난수로 된 `/data.txt` 파일을 만들 것입니다.
```vim
$ docker run -d ubuntu bash -c "shuf -i 1-10000 -n 1 -o /data.txt && tail -f /dev/null"
```
- 위 명령어는 bash shell을 실행하여 단일 난수를 `/data.txt`에 저장합니다. 그리고 컨테이너가 계속 실행되도록 파일을 감시합니다.

2. `exec` 터미널에 접속해줍니다. 접속하기 위해선 Docker Dashboard에서 `ubuntu` 이미지를 실행중인 컨테이너의 첫 번 째 작업(CLI)를 클릭해줍니다.
![img](/assets/img/docker/ubuntu.png)

- 그러면 ubuntu 컨테이너에서 실행가능한 터미널이 나타납니다.
![img](/assets/img/docker/exec.png)

- 아래의 명령어를 실행하여 `/data.txt` 파일의 내용을 확인해보세요.
```vim
$ cat /data.txt
```

- command line을 선호하는 경우 `docker exec` 명령어를 사용하여 동일한 작업을 수행 할 수 있습니다 . 컨테이너의 ID ( docker ps얻기 위해 사용) 를 얻고 다음 명령을 사용하여 내용을 가져와야합니다.

```vim
$ docker exec <container-id> cat /data.txt
```

임의의 숫자가 출력되어야 합니다.
![img](/assets/img/docker/capture02.png)

3. 이제 다른 ubuntu 컨테이너 (동일한 이미지)를 시작해 보겠습니다. 그러면 동일한 파일이 없다는 것을 알 수 있습니다.
```vim
$ docker run -it ubuntu ls /
```
![img](/assets/img/docker/capture03.png)
![img](/assets/img/docker/capture04.png)

4. `docker rm -f` 명령어를 실행해 ubuntu 컨테이너를 제거해줍니다.

## Container Volumes
이전 실험에서 각 컨테이너는 시작할 때마다 이미지 정의에서 시작하는 것을 확인했습니다. 

컨테이너는 파일을 생성, 업데이트 및 삭제할 수 있지만 컨테이너가 제거되고 모든 변경 사항이 해당 컨테이너에 격리되면 이러한 변경 사항이 손실됩니다. 

하지만 볼륨으로 이 모든 것을 변경할 수 있습니다.
볼륨은 컨테이너의 특정 파일 시스템 경로를 호스트 시스템에 다시 연결하는 기능을 제공합니다. 

컨테이너의 디렉토리가 마운트(하드디스크 또는 어떤 특정 장치를 인식하게 하는 것)되면 해당 디렉토리의 변경 사항도 호스트 시스템에서 볼 수 있습니다. 컨테이너 재시작시 동일한 디렉토리를 마운트하면 동일한 파일이 표시됩니다.

두 가지 주요 유형의 볼륨이 있습니다. 결국 둘 다 사용할 것이지만 **named volumes**부터 시작하겠습니다.

## Persisting our Todo Data
기본적으로 To do application은 `/etc/todos/todo.db`에서 [SQLite Database](https://www.sqlite.org/index.html)로 데이터가 저장됩니다.

SQLite는 모든 데이터가 단일 파일에 저장되는 단순한 관계형 데이터베이스입니다. 대규모 애플리케이션에는 적합하지 않지만 소규모 데모에서는 작동합니다. 나중에 다른 데이터베이스 엔진으로 전환하는 방법에 대해 설명하겠습니다.

해당 파일을 호스트에 유지하고 다음 컨테이너에서 사용할 수 있는 데이터베이스가 단일 파일인 경우, 마지막 파일이 중단된 위치를 선택할 수 있어야합니다. 

볼륨을 생성하고 데이터가 저장된 디렉토리에 연결(종종 "마운팅"이라고 함)함으로써 데이터를 유지할 수 있습니다. 컨테이너가 `todo.db` 파일에 쓸 때 볼륨의 호스트에 유지됩니다.

앞서 언급했듯이, **named volume**을 사용할 것 입니다. **named volume**은 간단한 데이터의 버킷이라고 생각하면 됩니다.

도커는 디스크의 물리적 위치를 유지해주며, 우리는 볼륨의 이름만 기억하면 됩니다. 볼륨을 사용할 때마다, 도커는 올바른 데이터가 제공되었는지 확인할 것 입니다.

1. `docker volume create` 명령어를 사용하여 볼륨을 만듭니다.
```vim
$ docker volume create todo-db
```

2. 지속적 볼륨을 사용하지 않고 계속 실행 중이므로 대시 보드 (또는 `docker rm -f <id>` 사용)에서 to do app container를 다시 한 번 중지해줍니다.

3. todo app container를 시작하되 볼륨 마운트를 지정하는 `-v` flag를 추가합니다. `/etc/todos` 로 마운트되고 named volume을 사용하면, 경로에 생성된 모든 파일이 캡처됩니다.
```vim
$ docker run -dp 3000:3000 -v todo-db:/etc/todos getting-started
```

4. 컨테이너가 시작되면 앱을 열고 to do list에 몇 가지 항목을 추가합니다.
![img](/assets/img/docker/todolist.png)

5. to do app의 컨테이너를 제거합니다.

6. 위와 동일한 명령을 사용하여 새 컨테이너를 시작해줍니다.

7. to do app을 엽니다. 목록에 여전히 항목이 표시되어야합니다!

8. 목록 확인이 끝나면 컨테이너를 제거해줍니다.

## Diving into our Volume
많은 사람들이 "named volume을 사용할 때 Docker가 내 데이터를 실제로 어디에 저장합니까?"라고 자주 질문합니다. 이를 알기 위해 `docker volume inspect`명령을 사용할 수 있습니다.

```json
$ docker volume inspect todo-db

[
    {
        "CreatedAt": "2019-09-26T02:18:36Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/todo-db/_data",
        "Name": "todo-db",
        "Options": {},
        "Scope": "local"
    }
]
```

`Mountpoint`는 데이터가 저장되어있는 디스크상의 실제 위치입니다.<br> 대부분의 장치는 호스트에서 이 디렉토리에 액세스하려면 루트 액세스 권한이 있어야합니다. 그러나 `Mountpoint`가 루트 액세스 권한이 있는 곳 입니다.

## Recap
우리는 이제 재시작 후에도 여전히 데이터가 유지되며 작동하는 application을 가지고 있습니다.

그러나 이전에 모든 변경 사항에 대해 이미지를 다시 작성하는데 시간이 많이 소요됨을 알게되었습니다. **bind mount**를 사용하면 더 쉽게 해결할 수 있습니다.


<br><br>

## Reference
1. http://localhost/tutorial/persisting-our-data/ 
