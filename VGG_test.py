# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:41:17 2019

@author: mandi_wang
"""
import numpy as np
from keras.layers import *
from keras.models import *
#from make_parallel import make_parallel
from keras import backend as K
from keras.callbacks import *
import numpy as np
import os
import cv2
from PIL import Image
import time
import sys
sys.path.append('./')

code=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#with open(r'../char_std_5990.txt','r',encoding='utf-8') as f:
#    for line in f:
#        code.append(line.strip('\n'))

with open('./dense_test/test_label.txt','r') as f:
    labelList=f.readlines()

    
model = load_model('vgg_best.h5')
#f = open('test_result.txt', 'w')

predict_dir = './dense_test/test_img/'

files = sorted([x for x in os.listdir(predict_dir) if '.jpg' in x])
n_files=len(files)
errors=0
total_time=0
def decode(pred):
    y = np.argmax(np.array(pred))
   # return ''.join([code[x] for x in y])
    print('概率值:',pred[0][y])
    if pred[0][y]<0.4:
        res = 16
    else:
        res=code[y]
    return res

for i in labelList:
    img = i.split(' ')[0]
    label = i.split(' ')[1].strip('\n')
    print(img,label)
 
    im = cv2.imread(predict_dir + img)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)[:,:,::-1].transpose(1, 0, 2)
    X = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    X[0] = im
  
    start_time = time.time()
    y_pred = model.predict(X)
   # print(y_pred.shape)
    end_time = time.time()
    time_dif=end_time-start_time
    y_pred_decode=decode(y_pred)
    print('预测：',y_pred_decode)
    
    print('cost time:%.3f ' %(time_dif))
    total_time+=time_dif
    
    ind=files.index(img)
    if  y_pred_decode!=int(label):
        errors += 1
        print('gt:',int(label))
        print('________Error________' )
        print('='*50)
   # else:
        
      #  print(''.join(y_pred_decode) + '=%d' % eval(''.join(y_pred_decode)))  #eval执行字符串表达式
   #     print(y_pred_decode)
   # f.write(''.join(y_pred_decode) + '\n')    

print('=' * 50)
print('Average Detection Time: %f' % (total_time/n_files))
print('=' * 50)
print('Precision: %d/%d' % (n_files-errors, n_files))
print('=' * 50)
#f.close()
