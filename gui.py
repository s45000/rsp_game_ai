import random as ra
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
image_size = 96
model = tf.keras.models.load_model(f'./good_model_{image_size}')

# 0,1,2 : 주먹, 가위, 보
label = ['주먹','가위','보']
# 승률 계산용
win_count = 0
game_count = 0

import cv2
# cap으로 영상을 불러옵니다.
cap = cv2.VideoCapture(0)
# 영상 프레임 사이즈를 결정합니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 250)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 220)

import tkinter as tk
from PIL import ImageTk, Image

# GUI 설계
win = tk.Tk() # 인스턴스 생성

win.title("RSPgame") # 제목 표시줄 추가
win.geometry("360x540+500+50") # 지오메트리: 너비x높이+x좌표+y좌표
win.resizable(False, False) # x축, y축 크기 조정 비활성화

# 라벨 추가
lbl = tk.Label(win, text="가위바위보 게임")
lbl.grid(row=0,column=0,columnspan=2)

# 프레임 추가
frm = tk.Frame(win, bg="white", width=720, height=480) # 프레임 너비, 높이 설정
frm.grid(row=1,column=0,columnspan=2) # 격자 행, 열 배치
# 라벨 추가 (웹캠 영상 출력용)
lblx = tk.Label(frm)
lblx.pack()
frame = None
def video_play():
    global frame
    ret, frame = cap.read() # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        cap.release() # 작업 완료 후 해제
        return
    iframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(iframe) # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img) # ImageTk 객체로 변환
    # OpenCV 동영상
    lblx.imgtk = imgtk
    lblx.configure(image=imgtk)
    lblx.after(10, video_play)

# 게임 진행 안내 텍스트
guideText = tk.Label(win, text = '게임시작을\n눌러주세요')
guideText.grid(row=2,column=0)

# 게임 진행 로그
gameLog = tk.Listbox(win)
gameLog.grid(row=2,column=1)

# 웹캠 이미지 캡처 -> 손 모양 분류
def handClassfication():
    # 현재 frame이 잘 처리되는지 확인하기 위해 저장
    cv2.imwrite("./lastshot.jpg",frame)
    # 저장된 프레임을 불러오며 input 형식에 맞게 변경
    img = image.load_img("./lastshot.jpg", target_size=(image_size, image_size))
    img = image.img_to_array(img)
    img = img.reshape((1,)+img.shape)
    img/255.0
    # 모델을 통한 예측
    res = model.predict(img)
    usr_hand = tf.argmax(res[0],0).numpy()
    return usr_hand

from threading import Thread
import time
end = 0
# 가위바위보 시작
def gameStart():
    global win_count,game_count,end
    cpu_hand = ra.randint(0,2) # 컴퓨터가 낼 것
    announce = ['가위..','바위..','보!!']
    # 가위.. 바위.. 보!! 타이밍에 맞춰 손 내기
    for i in range(2):
        guideText['text'] = announce[i]
        time.sleep(1.5)
    guideText['text'] = announce[2]
    # 보!! 했을때 사용자 화면 읽어와서 분류
    usr_hand = handClassfication()
    res = '' # 게임 결과
    # 가위바위보 결과 계산
    if cpu_hand == usr_hand: res = 'draw'
    elif cpu_hand-1 == usr_hand or (cpu_hand == 0 and usr_hand == 2):
        res = 'win'
        win_count += 1 # 이겼으면 승리 카운트 +1
    else: res = 'loss'
    game_count += 1
    # 컴퓨터가 낸 손 모양과 유저가 낸 손 모양 화면에 출력
    text = 'cpu : '+label[cpu_hand]+' usr : '+label[usr_hand]+f' {res}'
    gameLog.insert(end, text)
    end += 1 # 게임 결과를 하단에 이어붙이기 위해 end pointer +1
    # 승률 계산
    rate['text'] = f'승률 {round(100*win_count/game_count,2)}%'
    # 게임 로그 파일로 저장
    fdt = open('./game_log.txt','a')
    fdt.write(text+'\n')
    fdt.close()
    # 안내 문구 초기화
    guideText['text'] = '게임시작을\n눌러주세요'
    # 버튼 클릭 재 활성화
    b1['state'] = tk.NORMAL
    
#버튼 클릭
def lb1Click():
    b1['state'] = tk.DISABLED # 버튼 클릭 비활성화
    t1 = Thread(target = gameStart)
    t1.start() # 가위바위보 게임 시작

# 버튼 배치
b1 = tk.Button(win, text='게임 시작',command=lb1Click)
b1.grid(row=3,column=0)

# 라벨 배치
rate = tk.Label(win, text = 'Game Log')
rate.grid(row=3,column=1)

video_play()
win.mainloop() #GUI 시작
