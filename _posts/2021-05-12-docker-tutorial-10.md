---
layout: post
title: '[Docker] Part 10: What next'
subtitle: 'Docker Tutorial'
categories: devops
tags: doker
comments: true
---


[도커 공식 도큐먼트](https://docs.docker.com/get-started/)를 해석하면서 정리한 글입니다.

# Part 10. What Next?

## Container Orchestration
컨테이너가 어떻게 배치되서 동시에 실행할지에 대한 설정이라고 이해하면 됩니다. 앞선 예로 to do app 컨테이너와 MySQL 컨테이너를 생성하였는데, 이를 조합하여 실행해주는 것을 `Container Orchestration`이라 합니다.

이러한 기능을 해주는 도구로 Kubernetes, Swarm, Nomad 및 ECS가 있습니다.

이들의 특징으로 `expected state`와 `actual state`가 있습니다.

우리가 만들길 원하는 파일, 이를 `expected state`이며, 실제론 아직 아무것도 없는 `actual state`입니다. 쿠버네티스 등은 우리가 써놓은 명세를 보고 작업해주어 `expected state`와 `actual state`가 일치하게끔 해줍니다.

## Cloud Native Computing Foundation Projects

CNCF는 Kubernetes, Prometheus, Envoy, Linkerd, NATS 등을 포함한 다양한 오픈 소스 프로젝트를 위한 vendor-neutral home 입니다.

자세한 사항은 아래의 링크를 참조하시면 됩니다.
- https://www.cncf.io/projects/
- https://landscape.cncf.io/

이상으로 Docker tutorial을 끝마치겠습니다.

<br><br>

## Reference
1. http://localhost/tutorial/what-next/