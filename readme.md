# 파일별 간단 설명
save_npdata.py는 미리 설치된 데이터셋을 정제하여 .npy로 저장한다.
이를 위해 https://laurencemoroney.com/datasets.html의 Rock Paper Scissors Dataset을 설치해 사용한다.
training,valid,test 데이터셋을 모두 설치하고 save_npdata.py실행 위치에 rsp_data라는 폴더를 만들어
> rsp_data
> 1. paper
> 2. rock
> 3. scissors

형태가 되도록 하고, 각 폴더 내에 가위바위보 이미지들을 정리해둔 후 save_npdata.py를 실행시키면
train_data_imageSize.npy, train_label_imageSize.npy
valid_data_imageSize.npy, valid_label_imageSize.npy
test_data_imageSize.npy, test_label_imageSize.npy
파일이 생성된다. imageSize는 save_npdata.py 내부에서 확인

.npy파일들이 생성되면 train.py를 통해 모델을 학습(train_Log.txt, good_model_imageSize 생성)시키고,
test.py를 통해 테스트데이터 예측 및 모델 성능 지표를 확인(test_Log.txt 생성)할 수있다.

gui.py는 모델을 활용해 직접가위바위보를 할 수 있도록 하는 인터페이스이다.
실행시킬 컴퓨터의 웹캠이 동작하는 경우 정상적으로 사용가능하다.
game_Log.txt, lastshot.jpg(마지막으로 사용자가 낸 손) 생성

# 필요한 라이브러리
> from keras.preprocessing import image  
> import tensorflow  
> import numpy  
> from PIL import ImageFile  
> pip install opencv-python # 4.5.5.64  
> pip install pillow  
