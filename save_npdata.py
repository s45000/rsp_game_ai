import os
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # 이미지가 잘린경우 에러 방지

image_size = 96 # 이미지 사이즈 설정 (가로 세로 같은 길이)
label_len = 3 # 레이블 개수
names = ['rock','scissors','paper']

# 이미지 데이터를 약간씩 변화시켜 데이터 셋의 양을 늘리는 용도
datagen = image.ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# 데이터를 정제해 train, valid, test로 나눠담을 변수 선언
train_data = []
train_label = []
valid_data = []
valid_label = []
test_data = []
test_label = []

# 한 개의 이미지를 ImageDataGenerator를 사용해 늘릴 횟수
gen_num = 9
# 데이터셋이 정리된 폴더에 따라 'rock','scissors','paper' 한 레이블씩 데이터 정제
for label_index in range(3):
    data_path = './rsp_data/'+ names[label_index]
    print(data_path)
    pics = os.listdir(data_path)
    x = len(pics)
    # 폴더내의 데이터중 8/10은 train data
    for pic in pics[:int(x*(8/10))]:
        try:
            img=image.load_img(data_path+'/'+pic, target_size=(image_size, image_size))
            img = image.img_to_array(img)
            new_img = img.reshape((1,)+img.shape)
            idx = 0
            # 미리 선언해둔 datagen을 사용해 데이터 변화시키고 규격에 맞춘 후 저장
            for batch in datagen.flow(new_img, batch_size=1): # 여기서 batch는 new_img
                idx += 1
                train_data.append(batch[0]/255.0)
                train_label.append(label_index)
                if idx%gen_num == 0:
                    break
            # 원본 이미지도 저장
            img = img/255.0
            train_data.append(img)
            train_label.append(label_index)
        except:
            # 이미지에 문제가 있으면 에러처리
            print(data_path+'/'+pic+'is not available')
    # 폴더내의 데이터중 1/10은 valid data
    for pic in pics[int(x*(8/10)):int(x*(9/10))]:
        try:
            img=image.load_img(data_path+'/'+pic, target_size=(image_size, image_size))
            img = image.img_to_array(img)
            new_img = img.reshape((1,)+img.shape)
            idx = 0
            for batch in datagen.flow(new_img, batch_size=1):
                idx += 1
                valid_data.append(batch[0]/255.0)
                valid_label.append(label_index)
                if idx%gen_num == 0:
                    break
            valid_data.append(img/255.0)
            valid_label.append(label_index)
        except:
            print(data_path+'/'+pic+'is not available')
    # 폴더내의 데이터중 1/10은 test data 
    for pic in pics[int(x*(9/10)):int(x*(10/10))]:
        try:
            img=image.load_img(data_path+'/'+pic, target_size=(image_size, image_size))
            img = image.img_to_array(img)
            new_img = img.reshape((1,)+img.shape)
            idx = 0
            for batch in datagen.flow(new_img, batch_size=1):
                idx += 1
                test_data.append(batch[0]/255.0)
                test_label.append(label_index)
                if idx%gen_num == 0:
                    break
            img = img/255.0
            test_data.append(img)
            test_label.append(label_index)
        except:
            print(data_path+'/'+pic+'is not available')

# 데이터셋들 numpy로 변환     
train_data = np.array(train_data)
train_label = np.array(train_label)
train_label = tf.one_hot(train_label, label_len).numpy()
valid_data = np.array(valid_data)
valid_label = np.array(valid_label)
valid_label = tf.one_hot(valid_label, label_len).numpy()
test_data = np.array(test_data)
test_label = np.array(test_label)
test_label = tf.one_hot(test_label, label_len).numpy()

# 데이터셋들 npy파일로 저장
np.save(f'./train_data_{image_size}',train_data)
np.save(f'./train_label_{image_size}',train_label)
np.save(f'./valid_data_{image_size}',valid_data)
np.save(f'./valid_label_{image_size}',valid_label)
np.save(f'./test_data_{image_size}',test_data)
np.save(f'./test_label_{image_size}',test_label)
