# 과제에 명시된것 이외에 필요한 라이브러리
# pip install opencv-python # 4.5.5.64 (gui.py에서 사용)
# pip install pillow (gui.py, save_npdata.py에서 사용)


import tensorflow as tf
import numpy as np
import random as ra
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print('err')
        print(e)

# 모델 구조 출력
def getStrSummary(model):
    # 빈 리스트 준비
    lineList = []
    # model summary 수행
    # 단, 내용을 출력하는 대신 빈 리스트에 한 줄씩 추가함
    model.summary(print_fn=lambda x: lineList.append(x))
    # 리스트의 문자열들을 하나의 문자열로 합침
    return '\n'.join(lineList)

# input 이미지 가로 세로 픽셀 크기, label 개수
image_size = 96
label_len = 3

# 모델 구성
def my_model():
    model = tf.keras.Sequential() # layer를 하나씩 쌓아가기 위해 sequantial 모델 사용
    # 입력 layer : 이미지 가로 세로 픽셀을 가진 컬러이미지(96,96,3)
    model.add(tf.keras.layers.Input(shape=(image_size, image_size, 3)))
    # 입력값의 마지막 차원 기준으로 데이터 정규화 하여 학습에 사용
    model.add(tf.keras.layers.LayerNormalization(axis=-1))
    # 이미지 학습을 위한 합성곱 신경망
    model.add(tf.keras.layers.Convolution2D(64, 11,4 ,activation='relu'))
    # 학습 중간중간 연산결과를 정규화하여 일반화 성능을 높임
    model.add(tf.keras.layers.BatchNormalization())
    # 중요값만 추출해 데이터를 압축
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))

    # 데이터 형태를 크게 줄이지 않도록 0으로 둘러싸기
    model.add(tf.keras.layers.ZeroPadding2D((2,2)))
    model.add(tf.keras.layers.Convolution2D(256, 5, 1, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))
    
    model.add(tf.keras.layers.ZeroPadding2D((1,1)))
    model.add(tf.keras.layers.Convolution2D(256, 3, 1, activation='relu'))
    
    model.add(tf.keras.layers.ZeroPadding2D((1,1)))
    model.add(tf.keras.layers.Convolution2D(256, 3, 1, activation='relu'))
    
    model.add(tf.keras.layers.ZeroPadding2D((1,1)))
    model.add(tf.keras.layers.Convolution2D(256, 3, 1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))

    # Dense Layer 사용을 위한 1차원 데이터로 변경
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4)) # 과적합 방지로 랜덤 노드 무시하고 학습
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    # 출력 layer : label 수 (3), 출력값을 확률로 사용할 수 있는 softmax 활성함수 사용
    model.add(tf.keras.layers.Dense(label_len, activation='softmax'))

    return model
# 모델 생성
model = my_model()

# 학습 알고리즘: stochastic gradient descent
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.08, # 학습률
    momentum=0.9 # 모멘텀 계수
)
# 코스트(손실) 함수
loss = 'categorical_crossentropy'
# 성능 평가 척도: one-hot 레이블과 얼마나 일치하는지 정확도
metrics=['categorical_accuracy'] # label을 one-hot encoding한 후 사용
# 모델 컴파일
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# 모델 구조 요약
summary = getStrSummary(model)
print(summary)

# train, valid 데이터 불러오기
train_data = np.load(f'./train_data_{image_size}.npy')
train_label = np.load(f'./train_label_{image_size}.npy')
valid_data = np.load(f'./valid_data_{image_size}.npy')
valid_label = np.load(f'./valid_label_{image_size}.npy')
# train, valid 데이터 형태 출력
print(train_data.shape)
print(train_label.shape)
print(valid_data.shape)
print(valid_label.shape)

errorRate = 1.0
new_errorRate = 1.0
train_batch_size = 256 # 학습용 배치 사이즈
valid_batch_size = 256 # 검증용 배치 사이즈
itera = 0
train_log = '' # 저장할 학습 로그
start_time = time.time() # 학습 시작 시간
while round(errorRate * 100.0,2) > 5: # 에러율이 5% 미만이면 종료
    itera += 1
    # 미니 배치 학습을 위한 인덱스 섞기
    train_data_len = len(train_data)
    train_indices = list(range(train_data_len))
    ra.shuffle(train_indices)
    # valid 데이터 크기
    valid_data_len = len(valid_data)
    # train 데이터를 미니 배치 학습
    for batchCnt in range(int(train_data_len/train_batch_size)):
        trainLoss, trainAcc = model.train_on_batch(
                x=train_data[train_indices[batchCnt*train_batch_size:batchCnt*train_batch_size+train_batch_size]],
                y=train_label[train_indices[batchCnt*train_batch_size:batchCnt*train_batch_size+train_batch_size]]
            )
    # gpu용량제한으로 인해 evaluate 또한 배치로 진행 후 평균 계산
    # 평균을 계산하기 위해 배치별 evaluate값 담을 리스트 선언
    valid_loss = []
    valid_acc = []
    # valid 데이터 나눠서 성능 검증
    for batchCnt in range(int(valid_data_len/valid_batch_size)):
        loss, acc = model.evaluate(valid_data[batchCnt*valid_batch_size:batchCnt*valid_batch_size+valid_batch_size],
                                   valid_label[batchCnt*valid_batch_size:batchCnt*valid_batch_size+valid_batch_size], verbose=0)
        valid_loss.append(loss)
        valid_acc.append(acc)
    # valid_loss, valid_accuaracy 구하기
    validLoss = np.mean(valid_loss)
    validAcc = np.mean(valid_acc)
    # 학습로그 출력 및 파일로 저장할 로그에 누적
    text = f'train_loss : {trainLoss} train_acc : {trainAcc}'
    print(text)
    train_log += text+'\n'
    text = f'valid_loss : {validLoss} valid_acc : {validAcc}'
    print(text)
    train_log += text+'\n'
    cum_time = '('+str(round(time.time() - start_time,1))+'s)' # 지금까지 걸린 시간
    text = f'iterate {itera} {cum_time}'
    print(text)
    train_log += text+'\n\n'
    print()
    # 새로운 에러율 계산
    new_errorRate = 1.0 - validAcc
    # 새로운 에러율이 기존 에러율보다 높으면
    # errorRate 업데이트, 모델 저장
    if new_errorRate < errorRate:
        print('model saved')
        errorRate = new_errorRate
        model.save(f'./good_model_{image_size}')

# train 로그 파일로 저장
fdt = open('./train_log.txt','w')
fdt.write(train_log)
fdt.close()
