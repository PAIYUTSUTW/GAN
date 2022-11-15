#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# In[2]:

#from __future__ import absolute_import,print_function,division
from keras import layers
import keras
import keras.backend as K

import tensorflow as tf 
import tensorflow.keras as keras
import matplotlib.pyplot as plt 
import numpy as np 
import os
import PIL
import imageio
import glob
import time 

import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

from sklearn.utils import shuffle
import random


# In[3]:


width = 16
height = 256
channels = 1


# In[4]:


def readTrain():
  train = pd.read_csv("Driving Data test _3.csv")
  return train
def DriverA(data):
  dataA = data.iloc[0:7240,:]
  dataB = data.iloc[7241:20125,:]
  dataC = data.iloc[20126:27625,:]
  dataD = data.iloc[27626:40869,:]
  dataE = data.iloc[40870:49305,:]
  dataF = data.iloc[49306:60317,:]
  dataG = data.iloc[60318:67809,:]
  dataH = data.iloc[67810:77689,:]
  dataI = data.iloc[77690:85497,:]
  dataJ = data.iloc[85498:94401,:]
  return dataA,dataB,dataC,dataD,dataE,dataF,dataG,dataH,dataI,dataJ
def normalize(train):
  train = train.drop(["Class"], axis=1)
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm
def buildTrain(train, pastDay=60, futureDay=5):
  X_train, Y_train = [], []
  for i in range(train.shape[0]-height):
    X_train.append(np.array(train.iloc[i:i+height,0:width]))
    #Y_train.append(np.array(train.iloc[i:i+1]["Fuel_consumption"]))
  return np.array(X_train)
train = readTrain()
train_norm = normalize(train)


# In[5]:


dataA,dataB,dataC,dataD,dataE,dataF,dataG,dataH,dataI,dataJ = DriverA(train_norm)


# In[6]:


dataAA,dataBB,dataCC,dataDD,dataEE,dataFF,dataGG,dataHH,dataII,dataJJ = DriverA(train)


# In[7]:


trainA = buildTrain(dataA)
trainA = np.reshape(trainA,(trainA.shape[0],height,width,1))


# In[8]:


trainC = buildTrain(dataC)
trainC.shape
trainC = np.reshape(trainC,(trainC.shape[0],height,width,1))


# In[9]:


dataA_p = dataA.iloc[0:6000,0:width]
trainA_part = buildTrain(dataA_p)
trainA_part.shape
trainA_part = np.reshape(trainA_part,(trainA_part.shape[0],height,width,1))


# In[10]:


dataC_p = dataC.iloc[0:6000,0:width]
trainC_part = buildTrain(dataC_p)
trainC_part = np.reshape(trainC_part,(trainC_part.shape[0],height,width,1))


# In[11]:


dataAC_p = np.vstack([trainA_part,trainC_part])


# In[12]:


dataAC_p = np.reshape(dataAC_p,(dataAC_p.shape[0],width*height))


# In[13]:


labelA = np.ones(trainA_part.shape[0])
labelC = np.zeros(trainC_part.shape[0])
labelAC = np.concatenate([labelA,labelC])


# In[14]:


index = [i for i in range(len(dataAC_p))]
random.shuffle(index)
dataAC_p = dataAC_p[index]
labelAC = labelAC[index]


# In[15]:


dataAC_p_r = np.reshape(dataAC_p,(dataAC_p.shape[0],height,width,1))


# In[16]:


dataAA_p = dataA.iloc[0:7000,0:width]
trainAA_part = buildTrain(dataAA_p)
trainAA_part.shape
trainAA_part = np.reshape(trainAA_part,(trainAA_part.shape[0],height,width,1))
labelAA = np.zeros(trainAA_part.shape[0])


# In[17]:


dataCC_p = dataC.iloc[0:7000,0:width]
trainCC_part = buildTrain(dataCC_p)
trainCC_part.shape
trainCC_part = np.reshape(trainCC_part,(trainCC_part.shape[0],height,width,1))
labelBB = np.zeros(trainCC_part.shape[0])


# In[18]:


#Generators Encoder
input_layer = layers.Input(name='input', shape=(height, width, channels))

# Encoder
x = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
x = layers.LeakyReLU(name='leaky_1')(x)

x = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(name='leaky_2')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_3', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(name='leaky_3')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_4', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(name='leaky_4')(x)

x = layers.Conv2D(256, (3,3), strides=(2,2), padding='same', name='conv_5', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_4')(x)
x = layers.LeakyReLU(name='leaky_5')(x)

x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)

g_e = keras.models.Model(inputs=input_layer, outputs=x)

g_e.summary()


# In[19]:


#Generator
input_layer = layers.Input(name='input', shape=(height, width, channels))

x = g_e(input_layer)

y = layers.Dense((height/8)*(width/8)*256, name='dense')(x) # 2 = 128 / 8 / 8
y = layers.Reshape((height//8, width//8,256), name='de_reshape')(y)

#y = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same', name='deconv_0', kernel_regularizer = 'l2')(y)
#y = layers.LeakyReLU(name='de_leaky_0')(y)

y = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', name='deconv_1', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_1')(y)

y = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='deconv_2', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_2')(y)

y = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='deconv_3', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_3')(y)

y = layers.Conv2DTranspose(channels, (1, 1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer = 'l2', activation='tanh')(y)

g = keras.models.Model(inputs=input_layer, outputs=y)

g.summary()


# In[20]:


#Encoder
input_layer = layers.Input(name='input', shape=(height, width, channels))

z = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='encoder_conv_1', kernel_regularizer = 'l2')(input_layer)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='encoder_conv_2', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_1')(z)
z = layers.LeakyReLU()(z)


z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='encoder_conv_3', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_2')(z)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(256, (3,3), strides=(2,2), padding='same', name='encoder_conv_4', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_3')(z)
z = layers.LeakyReLU()(z)


z = layers.Conv2D(256, (3,3), strides=(2,2), padding='same', name='conv_41', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_4')(z)
z = layers.LeakyReLU()(z)

z = layers.GlobalAveragePooling2D(name='encoder_output')(z)

encoder = keras.models.Model(input_layer, z)
encoder.summary()


# In[21]:


#feature extractor
input_layer = layers.Input(name='input', shape=(height, width, channels))

f = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='f_conv_1', kernel_regularizer = 'l2')(input_layer)
f = layers.LeakyReLU(name='f_leaky_1')(f)

f = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='f_conv_2', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_1')(f)
f = layers.LeakyReLU(name='f_leaky_2')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_3', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_2')(f)
f = layers.LeakyReLU(name='f_leaky_3')(f)

f = layers.Conv2D(256, (3,3), strides=(2,2), padding='same', name='f_conv_4', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_3')(f)
f = layers.LeakyReLU(name='f_leaky_4')(f)

f = layers.Conv2D(256, (3,3), strides=(2,2), padding='same', name='f_conv_5', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_4')(f)
f = layers.LeakyReLU(name='feature_output')(f)

feature_extractor = keras.models.Model(input_layer, f)

feature_extractor.summary()


# In[22]:


x[1]


# In[23]:


#gan trainer
class AdvLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class CntLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - encoder(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
# model for training
input_layer = layers.Input(name='input', shape=(height, width, channels))
gan = g(input_layer) # g(x)

adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss])

# loss function
def loss(yt, yp):
    return yp

losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
}

lossWeights = {'cnt_loss': 20.0, 'adv_loss': 1.0, 'enc_loss': 1.0}

# compile
gan_trainer.compile(optimizer = 'adam', loss=losses, loss_weights=lossWeights)


# In[24]:


gan_trainer.summary()


# In[25]:


#discriminator

input_layer = layers.Input(name='input', shape=(height, width, channels))

f = feature_extractor(input_layer)

d = layers.GlobalAveragePooling2D(name='glb_avg')(f)
d = layers.Dense(1, activation='sigmoid', name='d_out')(d)
    
d = keras.models.Model(input_layer, d)
d.summary()


# In[26]:


d.compile(optimizer='adam', loss='binary_crossentropy')


# In[27]:


#Training

niter = 1500
bz = 32
def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y]        


# In[28]:


train_data_generator = get_data_generator(trainC, bz)


# In[ ]:


for i in range(niter):
    
    ### get batch x, y ###
    x, y = train_data_generator.__next__()
        
    ### train disciminator ###
    d.trainable = True
        
    fake_x = g.predict(x)
        
    d_x = np.concatenate([x, fake_x], axis=0)
    d_y = np.concatenate([np.zeros(len(x)), np.ones(len(fake_x))], axis=0)
        
    d_loss = d.train_on_batch(d_x, d_y)

    ### train generator ###
    
    d.trainable = False        
    g_loss = gan_trainer.train_on_batch(x, y)
    
    if i % 50 == 0:
        print(f'niter: {i+1}, g_loss: {g_loss}, d_loss: {d_loss}')


# In[29]:


#Evaluation
encoded = g_e.predict(dataAC_p_r)
gan_x = g.predict(dataAC_p_r)
encoded_gan = g_e.predict(gan_x)
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
score_simple = np.sum(score,axis = 0)
score_simple=score_simple/len(score)


# In[30]:


score_simple


# In[32]:


score = np.reshape(score,(1,score.shape[0]))


# In[33]:


score_binarizer = Binarizer(threshold=0.05).fit(score)


# In[34]:


binaryscore = score_binarizer.transform(score)


# In[35]:


labelAC_t = np.reshape(labelAC,(1,score.shape[1]))


# In[36]:


(((score.shape[1])-(binaryscore!=labelAC).sum())/score.shape[1])


# In[37]:


from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 5
plt.scatter(range(len(dataAC_p)),score, c=['skyblue' if x == 0 else 'pink' for x in labelAC])


# In[ ]:





# In[ ]:




