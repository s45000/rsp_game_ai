import tensorflow as tf
import numpy as np

# inputdata 가로세로 픽셀, label 개수
image_size = 96
label_len = 3

# 테스트 데이터 불러오기
test_data = np.load(f'./test_data_{image_size}.npy')
test_label = np.load(f'./test_label_{image_size}.npy')

# 테스트 데이터 형태 출력
print(test_data.shape)
print(test_label.shape)

# 사용할 모델 불러오기
model = tf.keras.models.load_model(f'./good_model_{image_size}')

test_log = '' # 저장할 로그
err,count = 0,0 # 에러율 계산용
d = {i : [0,0,0] for i in range(3)} # 각 레이블별 맞춘수, 전체수, 이label이라고 예측한수
for p, a in zip(model.predict(test_data),test_label): # 테스트 데이터를 예측후 결과와 한개씩 맞춰보기
    x = tf.argmax(p,0).numpy()
    y = tf.argmax(a,0).numpy()
    # 에러율 계산용
    if x != y: err += 1
    count += 1
    # 성능평가를 위한 각 label별 지표 누적
    if x == y:
        d[y][0]+=1 # 예측과 정답이 일치한 개수
    d[y][1]+=1 # 정답의 개수
    d[x][2]+=1 # 예측한 개수
    # 데이터 한개에 대한 로그 출력 및 로그 누적
    log = '예측 : '+str(x)+' 정답 : '+str(y)+' 오류율 : '+ str(round(err/count*100,2))+'%'
    test_log += log+'\n'
log = '맞춘개수\t전체개수\t예측개수'
test_log += log+'\n'
label = ['주먹','가위','보']
# 각 레이블별 맞춘개수, 전체개수, 예측개수
for i in range(3):
    log = str(d[i][0])+'\t'+str(d[i][1])+'\t'+str(d[i][2])+'\t'+label[i]
    test_log += log+'\n'
# 모델의 성능 평가
test_log += '모델 성능 평가\n'
# Accuracy (정확도)
test_log += f'Accuracy : {round(100*(d[0][0]+d[1][0]+d[2][0])/(d[0][1]+d[1][1]+d[2][1]),2)}\n'
# Precision (정밀도)
for i in range(3):
    test_log += f'Precision {label[i]} : {round(d[i][0]/d[i][2],2)}\n'
# Recall (재현율)
for i in range(3):
    test_log += f'Recall {label[i]} : {round(d[i][0]/d[i][1],2)}\n'
# F1-score : 2*(Precision*Recall)/(Precision+Recall)
for i in range(3):
    test_log += f'F1-score {label[i]} : {round(2*((d[i][0]/d[i][2])*(d[i][0]/d[i][1]))/(d[i][0]/d[i][2]+d[i][0]/d[i][1]),2)}\n'
print(test_log)
# 로그 저장
fdt = open('./test_log.txt','w')
fdt.write(test_log+'\n')
fdt.close()

    
