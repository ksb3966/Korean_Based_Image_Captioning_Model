# Korean_Based_Image_Captioning_Model
# 한국어 기반 이미지 캡셔닝 모델 구현

Image Captioning 기술은 이미지 데이터에 Caption, 즉 그 이미지에 대한 설명을 달아주는 기술이다. 이는 이미지의
물체나 동작 등을 인식하는 컴퓨터 비전 기술, 그리고 이를 통해 찾아낸 단어 토큰들을 하나의 문장으로 엮어내는
자연어 처리 기술의 집합이라고 할 수 있다. 이 기술은 영어를 사용하는 여러 국가들 사이에서는 활발히 연구되고
있고, 매년 새로운 논문들과 기술들이 발표되고 있는 유망한 분야이다. 하지만, 대부분의 논문들과 기술들이 영어
기반으로 구성되어 있으며, 한국어를 기반으로 한 Image Captioning 연구는 아직 미진한 상황이다. 이에 한국어
기반 Image Captioning 모델을 구현해보고 이를 제시한다. 

이 리포지토리는 'Show, Attend and Tell' 논문(https://arxiv.org/pdf/1502.03044.pdf) 을 기반으로 한다. 본 논문에서는 이미지를 Encoding하고 Feature를 추출하는 과정에서 CNN을, 그리고 Encoding한 이미지를 단어 단위로 추출해 문장을 형성하는 과정에서 RNN을 사용하게 된다. 가장 먼저 (1)Inception V3과 LSTM으로 구성된 모델과, (2)ResNet-101과 LSTM으로 구성된 모델, (2)ResNet-101과 KoBert로 이루어진 모델 총 3가지로 실험해보았다. 

## Instructions to run the Code 
### InceptionV3 & Resnet-101 + LSTM Model
1. ipynb 파일을 열고, 그 파일에 annotation file의 경로를 설정해준다.
2. ipynb을 실행시킨다.

### ResNet-101 + KoBert
#### 데이터 다운로드 및 전처리 과정
1.	3개의 폴더를 만듭니다. (1) data, (2) annotations - inside of data, (3) checkpoints
2.	MS COCO2014 데이터셋을 다운로드하고, (1) data 폴더 안에 압축해제한 후 저장합니다.
3.	(1) data > annotation를 만들고, folder 안에  annotation file(AI Hub의 Korean caption data)을 저장합니다.
4.	processData.py 파일을 실행시킵니다. - train2014_resized, val2014_resized, kor_vocab.pkl 파일 생성
5.	processData.py의 마지막 줄을 주석 처리합니다.

#### 모델 훈련 및 평가
1.	train_korbert.py 파일을 열고 실행시킵니다.
2.	validation_korbert.py 파일을 열고 실행시킵니다.

#### 모델 다운로드 링크
1.	MS COCO train2014 : http://images.cocodataset.org/zips/train2014.zip
2.	MS COCO val2014 : http://images.cocodataset.org/zips/val2014.zip
