# Generative-Adversarial-Networks
(22/07 ~ 23/02) 광운대학교 산학연계 프로젝트 - Team 사미용두

## Introduction
Apply Generative Adversarial Nets to generating leak data.

## 주요 기능
1. 리크 데이터 생성
>* GAN 모델로 랜덤 생성

2. 불량 이미지 생성
>* 생성 위치 랜덤 지정
>* 생성 위치 사용자 지정
>* 여러개 리크 생성

3. 불량 이미지 다운로드

## 서비스 화면

### 메인 화면

![메인화면](https://user-images.githubusercontent.com/49435654/214221161-7d22fda2-fc1e-437f-8203-15548903a60d.png)

### 사용자 설정 화면

![사용자지정화면](https://user-images.githubusercontent.com/49435654/214539567-062acd1a-b716-4b74-b032-76c2cd566987.png)

### 불량 이미지 생성 화면

![불량 이미지 생성 화면](https://user-images.githubusercontent.com/49435654/214539519-1c42ed91-1b0f-4b55-ae23-2ec2cc26d761.png)

## 시스템 구성 및 아키텍쳐

### Back-End & Front-End

><img width="586" alt="BackEnd   FrontEnd" src="https://user-images.githubusercontent.com/49435654/214540132-9de4ccb3-9f93-4d88-913d-e7fca6e7e054.png">



### Modeling
> 1. Pre-processing
>> ![데이터 전처리 아키텍쳐(업데이트 필요)](https://user-images.githubusercontent.com/49435654/214227127-d231629b-cd5d-4150-ab03-8c82ebaaff55.PNG)

> 
> 2. Modeling
>> 출처: [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
>> ![dcgan구조](https://user-images.githubusercontent.com/49435654/214542274-3ed33377-1ba8-4a20-ac94-8a17dd7ed414.png)



> 3. Generate Defective Image
>> ![불량이미지 생성 아키텍쳐(업데이트 필요)](https://user-images.githubusercontent.com/49435654/214229041-a5a0402b-aa18-4c9f-b651-fb126d86e511.PNG)


## Requirements:
* Python 3.9.5
* OpenCV 4.6.0
* Pytorch 1.12.1

## Prerequistes

node.js & npm

## checkout, build & run

```
1. git clone
2. npm install 
4. npm run start
```

## Contact Us
* [김대현](https://github.com/DevDae)
* [김재윤](https://github.com/kimjaeyoonn)
* [김태영](https://github.com/kty4119)
* [손재성](https://github.com/noseaj)
* [장효영](https://github.com/HyoYoung22)
