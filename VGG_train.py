# -*- coding: utf-8 -*-
import tensorflow as tf
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
import cv2
import math
import numpy as np
import os

#os.environ["PATH"] += os.pathsep + 'D:/graphviz-2.38/release/bin/'
#from tensorflow.keras.utils import plot_model  #可视化
#from IPython.display import Image

width=224
height=224
n_class=16
batch_size=64
img_dir=r'./data/img/'
with open(r'./data/label.txt','r') as f:
    label=f.read().splitlines()
num_files=len(label)
img_set = np.zeros((num_files, width, height, 3), dtype=np.uint8)
label_set = [np.zeros((num_files, n_class), dtype=np.uint8)for i in range(n_len)]  #4个数组
label_len_set = np.zeros((num_files, 1), dtype=np.int32)

def gen(inputs=None, targets=None, label_length_train=None, batch_size=None, shuffle=False): 

    while True:
        if shuffle:
            #inputs = tf.random_shuffle(inputs)
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                #inputs = tf.strided_slice(inputs,start_idx,start_idx + batch_size,)
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            #yield inputs, targets[excerpt]
            targets_label=[] 
            for k in range(n_len):
                targets_k=targets[k][excerpt][:]
                targets_label.append(targets_k)          
            yield inputs[excerpt], targets_label

for i in range(num_files):
    print(i)
    imgname=label[i].split(' ')[0]
    img=cv2.imread(img_dir+imgname) #h,w,c
    img=cv2.resize(img,(width,height))
   # train_set[i] = img[:,:,::-1].transpose(1,0,2)
    img_set[i]=img.transpose(1,0,2) 
    label_line=label[i]  
    str_label_line=label_line.split(' ')
    change_label=int(str_label_line[1])
    for j in range(len(label_set)):
        label_set[j][i][change_label]=1
train_len=math.ceil(num_files*0.9)
test_len=num_files-train_len
train_set=img_set[:train_len,...]
test_set=img_set[train_len:,...]
train_label=[]
test_label=[]
for k in range(n_len):
    train_label_k=label_set[k][:train_len][:]
    test_label_k=label_set[k][train_len:][:]
    train_label.append(train_label_k)
    test_label.append(test_label_k)

# 输入层
inputs = Input(shape=(width, height, 3))

# 卷积层和最大池化层
conv1 = Conv2D(4, (3,3), padding='same', activation='relu')(inputs)
conv2 = Conv2D(4, (3,3), padding='same', activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=2)(conv2)

conv3 = Conv2D(8, (3,3), padding='same', activation='relu')(pool1)
conv4 = Conv2D(8, (3,3), padding='same', activation='relu')(conv3)
pool2 = MaxPooling2D(pool_size=2)(conv4)

conv5 = Conv2D(16, (3,3), padding='same', activation='relu')(pool2)
conv6 = Conv2D(16, (3,3), padding='same', activation='relu')(conv5)
conv7 = Conv2D(16, (3,3), padding='same', activation='relu')(conv6)
pool3 = MaxPooling2D(pool_size=2)(conv7)

conv8 = Conv2D(32, (3,3), padding='same', activation='relu')(pool3)
conv9 = Conv2D(32, (3,3), padding='same', activation='relu')(conv8)
conv10 = Conv2D(32, (3,3), padding='same', activation='relu')(conv9)
pool4 = MaxPooling2D(pool_size=2)(conv10)

conv11 = Conv2D(64, (3,3), padding='same', activation='relu')(pool4)
conv12 = Conv2D(64, (3,3), padding='same', activation='relu')(conv11)
conv13 = Conv2D(64, (3,3), padding='same', activation='relu')(conv12)
pool5 = MaxPooling2D(pool_size=2)(conv13)

# 扁平层
flat = Flatten()(pool5)

# 全联接层
fc1 = Dense(256, activation='relu')(flat)
fc2 = Dense(256, activation='relu')(fc1)

# 输出层
outputs = Dense(n_class, activation='softmax')(fc2)

my_vggmodel = Model(inputs=inputs, outputs=outputs)
my_vggmodel.summary()
#plot_model(model, to_file='cnn.png', show_shapes=True)  #可视化
#Image('cnn.png')

callbacks = [EarlyStopping(patience=3), CSVLogger('vgg.csv'), ModelCheckpoint('vgg_best.h5', save_best_only=True)]
my_vggmodel.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-3, amsgrad=True), 
              metrics=['accuracy'])
my_vggmodel.fit_generator(gen(train_set, train_label,batch_size=batch_size,shuffle=True), epochs=100, validation_data=(test_set, test_label), workers=4,use_multiprocessing=True,
                    steps_per_epoch=round(num_files/batch_size),
                    callbacks=callbacks)

