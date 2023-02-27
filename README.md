# Generative-Adversarial-Networks
* (2022-07 ~ 2023-02) 광운대학교 산학연계 SW프로젝트 : 사미용두팀

## Introduction
자동차 부품 제조 현장의 학습/검증용 데이터 불균형을 해결하기 위한 GAN 기반의 불량 데이터 생성

## Directory Structure
```
SMYD
|- data
| |- L_01-1 ~ L_01-6  # Any images
|- models
| |- generator.py
| |- discriminator.py
| |- train.py
|- utils
| |- preprocess.py
| |- dataloader.py
|- results
| |- scoring.py
|- main.py
|- requirements.txt
| ...

```
# Project Information
## Requirements
* Python 3.9.5
* OpenCV 4.6.0.66
* Pytorch 1.12.1
* Torchvision 0.13.1
* Numpy 1.22.3
* Scipy 1.7.1

## Prerequistes
* node.js
* npm
* nodemon

## Build & Run
```
1. git clone
2. npm install 
3. npm run start
```
##  구현 기능  
### 1. Pre-processing  
> 사전 데이터  
>   ↓  
> (1) 리크 추출  
> (2) 그라드라인 제거  
> (3) 리사이즈  
> (4) 리크 데이터 증강  
>  ↓  
> 학습 데이터  

### 2. Modeling
> ![dcgan](https://user-images.githubusercontent.com/88760905/221587571-b73fd2b8-d170-4f2b-b5d0-a80e6b761645.png)  
> 출처 : [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)

### 3. Generate Defective Image  
> 정상 데이터  
>   ↓  
> (1) 모델 불러오기  
> (2) 리크 생성  
> (3) 사용자 설정 확인  
> (4) 정상 데이터에 합성  
>  ↓  
> 불량 데이터 

# Project Results
## 사용 방법  
### 1. 이미지 삽입
> 이미지 파일 업로드

### 2. 리크 세부 설정
> 생성 위치 랜덤 지정
> 생성 위치 사용자 지정
> 여러개 리크 생성

### 3. 이미지 생성 및 저장
> 불량 이미지 다운로드  
 
## 서비스 화면
|  메인 화면   |  사용자 설정 화면  |  불량 이미지 생성 화면  |
| :--------: | :------------: | :----------------: |
| ![메인화면](https://user-images.githubusercontent.com/49435654/214221161-7d22fda2-fc1e-437f-8203-15548903a60d.png) | ![사용자지정화면](https://user-images.githubusercontent.com/49435654/214539567-062acd1a-b716-4b74-b032-76c2cd566987.png) | ![불량 이미지 생성 화면](https://user-images.githubusercontent.com/88760905/221583943-5a6a41db-6668-4aaf-a399-174f43a7dfc7.png) |

## 시스템 구성
| JavaScript |    CSS    |    HTML    |   Node.JS  |   Express  |  Python  |  Pytorch  |
| :--------: | :-------: | :--------: | :--------: | :--------: | :------: |  :------: |
| ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E.svg?&style=flat&logo=JavaScript&logoColor=black) | ![CSS3](https://img.shields.io/badge/CSS3-1572B6.svg?&style=flat&logo=CSS&logoColor=white) | ![HTML](https://img.shields.io/badge/HTML5-E34F26.svg?&style=flat&logo=HTML&logoColor=white) | ![NodeJS](https://img.shields.io/badge/node.js-339933?style=flat&logo=NODE.JS&logoColor=white) | <img src="https://img.shields.io/badge/express-000000?style=flat&logo=EXPRESS&logoColor=white"> | ![Python](https://img.shields.io/badge/Python-3776AB.svg?&style=flat&logo=Python&logoColor=white) | ![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?&style=flat&logo=Pytorch&logoColor=white) |

# Team SMYD
### Contact Us
* [김대현](https://github.com/DevDae) | [김재윤](https://github.com/kimjaeyoonn) | [김태영](https://github.com/kty4119) | [손재성](https://github.com/noseaj) | [장효영](https://github.com/HyoYoung22)
