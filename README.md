# 얼굴/번호판 비식별화 프로그램
## 모델 및 기술 소개
#### [기존 기술의 문제점]
- [기존 오픈소스 모델](https://github.com/yeephycho/tensorflow-face-detection) 사용시 작은 물체들을 인식하지 못합니다
#### [해결방법]
- 실제 한국에서 촬영된 약 4만장의 사진들을 YOLO v3 모델로 학습시켜 작은 물체를 효과적으로 찾아낼 수 있습니다<br>
#### YOLO v3 모델 소개
대표적인 Real-Time Object Detection의 state-of-the-art 모델로 COCO 데이터셋을 학습시 30FPS 에 mAP 57.9% 의 성능을 자랑합니다
<p align="center">
  <img src="./contents/yolov3.jpg" width="450" title="performance_image">
</p>

## 사용법
1. 다음 링크를 통해 Anaconda 환경을 설치해줍니다. [설치 링크](https://docs.anaconda.com/anaconda/install/windows/)
2. 다음 명령어를 통해 새로운 Anaconda 가상 환경을 만들고 Dependency 를 설치합니다.
    2.1. 리포지토리 위치로 이동
    ```bash
    cd [absolute_path_to_repository]
    ```
    2.2. 패키지 설치
    ```bash
    conda install tensorflow==1.14
    ```
    ```bash
    conda install pytorch torchvision cpuonly -c pytorch
    ```
    ```bash
    conda install -c menpo opencv
    ```
    ```bash
    pip install opencv-contrib-python
    ```
    ```bash
    pip install --upgrade lxml
    ```
    ```bash
    conda install matplotlib
    ```
    
3. 얼굴/번호판 인식을 하고자 하는 사진을 ```input``` 폴더에 다음과 같은 파일 구조로 붙여넣습니다.
    ```
    Project Directory
        ├──input
        │   ├── folder_1
        │	│     ├── image_1.jpg
        │	│	  ├── image_2.jpg
        │	│	  ├── ...
        │   │
        │   ├── folder_2
        │   └── ...
    ```
4. 다음 명령어를 통해 프로그램을 실행시킵니다.
    ```bash
    python main.py
    ```
5. ```ouput``` 폴더에 결과가 출력됩니다.

## 성능
Class | AP
------------ | -------------
Face | 38.43
License Plate | 86.08

Model | MAP
------------ | -------------
Our Model | 62.25
