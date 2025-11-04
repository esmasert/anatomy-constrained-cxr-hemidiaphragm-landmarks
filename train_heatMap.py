import gc
gc.collect()

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
if ram_gb < 20:
    print('Not using a high-RAM runtime')
else:
    print('You are using a high-RAM runtime!')

#Imports
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torchvision
from keras.layers import *
from keras.models import *
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from PIL import Image
import heapq
import tensorflow as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Read in data
csvFilePath="/data/HeatMap/AllDataFinalVersion.csv"
df = pd.read_csv(csvFilePath)
df.fillna(method = 'ffill',inplace = True)

print(len(df))

#Transform the data into images and plot one to see
imgs = []
for i in range(0,len(df)):
    img = df['image_list'][i].split()
    img = ['0' if x == '' else x for x in img]
    imgs.append(img)

print(len(imgs))

image_list = np.array(imgs,dtype = 'float')
print(image_list.shape)

X_train = image_list.reshape(-1,512,512,1)
print(X_train.shape)

#Get the keypoint labels
training = df.drop('image_list',axis = 1)

y_train = []
for i in range(0,len(df)):
    y = training.iloc[i,:]
    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')

print(X_train.shape)
print(y_train.shape)

# Step 2: Creating the Heatmaps
"""
Okay, so like I said before, we are going to be predicting heatmaps with our model. So how do we actually get our labelled heatmap data? Turns out it is actually quite easy, we simply pass our Cartesian coordinates through a 2D gaussian kernel.
"""

#Function to create heatmaps by convoluting a 2D gaussian kernel over a (x,y) keypoint.
def gaussian(xL, yL, H, W, sigma=5):

    channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))

    return channel

#Generate heatmaps for one sample image
heatmaps = []

for i in range(0, 2, 1):
    x = int(y_train[0][i])
    y = int(y_train[0][i + 1])
    heatmap = gaussian(x, y, 512, 512)
    heatmaps.append(heatmap)

heatmaps = np.array(heatmaps)
print(heatmaps.shape)

# Step 3: Keras Custom Generator

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, imgs, kps, batch_size=1, shuffle=True):
        self.imgs = imgs
        self.kps = kps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.imgs) // self.batch_size

    def __getitem__(self, index):
        #Get index of images to generate
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        #Shuffle the data after the generator has run through all samples
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def gaussian(self, xL, yL, H, W, sigma=5):
        ##Function that creates the heatmaps##
        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X_batch = [self.imgs[i] for i in indexes]
        X_batch = np.array(X_batch)

        y_batch = []

        kps = [self.kps[i] for i in indexes]

        for i in range(0,len(kps)):
            heatmaps = []
            for j in range(0, 2):
                x = int(kps[i][j*2])
                y = int(kps[i][j*2+1])
                heatmap = self.gaussian(x, y, 512, 512)
                heatmaps.append(heatmap)
            y_batch.append(heatmaps)

        y_batch = np.array(y_batch)
        y_batch = np.swapaxes(y_batch,1,3)
        y_batch = np.swapaxes(y_batch,1,2)
        return X_batch, [y_batch, y_batch]

#Testing to see if our DataGenerator is working
X_batch, [y_batch, _] = next(DataGenerator(X_train, y_train).__iter__())
print(X_batch.shape)
print(y_batch.shape)

# Step 4: Creating the Model

#Helper function for building model
def conv_block(x, nconvs, n_filters, block_name, wd=None):
    for i in range(nconvs):
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   kernel_regularizer=wd, name=block_name + "_conv" + str(i + 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name=block_name + "_pool")(x)
    return x

#Represents one stage of the model
def stages(x, stage_num, num_keypoints = 2):

    #Block 1
    x = conv_block(x, nconvs=2, n_filters=64, block_name="block1_stage{}".format(stage_num))

    #Block 2
    x = conv_block(x, nconvs=2, n_filters=128, block_name="block2_stage{}".format(stage_num))

    #Block 3
    pool3 = conv_block(x, nconvs=3, n_filters=256, block_name="block3_stage{}".format(stage_num))

    #Block 4
    pool4 = conv_block(pool3, nconvs=3, n_filters=512, block_name="block4_stage{}".format(stage_num))

    #Block 5
    x = conv_block(pool4, nconvs=3, n_filters=512, block_name="block5_stage{}".format(stage_num))

    #Convolution 6
    x = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv6_stage{}".format(stage_num))(x)

    #Convolution 7
    x = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv7_stage{}".format(stage_num))(x)

    #upsampling
    preds_pool3 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool3_stage{}".format(stage_num))(pool3)
    preds_pool4 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool4_stage{}".format(stage_num))(pool4)
    up_pool4 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation='relu', name="ConvT_pool4_stage{}".format(stage_num))(preds_pool4)
    up_conv7 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation='relu', name="ConvT_conv7_stage{}".format(stage_num))(x)

    fusion = Add()([preds_pool3, up_pool4, up_conv7])

    heatmaps = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu', name="convT_fusion_stage{}".format(stage_num))(fusion)
    heatmaps = Conv2D(num_keypoints, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output_stage{}".format(stage_num))(heatmaps)
    return heatmaps

#Create a single stage FCN
def build_model(input_shape):
    outputs = []

    img = Input(shape=input_shape, name="Input_stage")

    ### Stage 1 ###
    heatmaps1 = stages(img, 1)
    outputs.append(heatmaps1)

    ### Stage 2 ###
    x = Concatenate()([img, heatmaps1])
    heatmaps2 = stages(x, 2)
    outputs.append(heatmaps2)

    model = Model(inputs=img, outputs=outputs, name="FCN_Final")
    return model

#Training the model using mean squared losss
def get_loss_func():
    def mse(x, y):
        return mean_squared_error(x,y)

    keys = ['output_stage1', 'output_stage2']
    losses = dict.fromkeys(keys, mse)
    return losses

model = build_model((512,512,1))
losses = get_loss_func()
model.compile(loss = losses, optimizer = 'adam', metrics=[tf.keras.metrics.Accuracy()])
model.summary()


#First, lets split the data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

#Create your two generators for train and validation
gen_train = DataGenerator(X_train, y_train)
gen_val = DataGenerator(X_val, y_val)

filepath="/data/HeatMap/models/weights-improvement-{epoch:02d}.hdf5"

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
savehistory = tf.keras.callbacks.CSVLogger('/data/HeatMap/history.csv')
callbacks_list = [checkpoint,savehistory]

#Train the model
history = model.fit_generator(generator = gen_train,
                    epochs = 200,
                    validation_data = gen_val,
                    callbacks=callbacks_list)


print("Model is saving...")
model.save('/data/HeatMap/models/')

print(history.history.keys())

print("Graphs are saving...")
plt.plot(history.history['output_stage2_accuracy'])
plt.plot(history.history['val_output_stage2_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train stage2 accuracy', 'val stage2 accuracy'], loc='upper left')
#plt.show()
plt.savefig('/data/HeatMap/graphs/train_acc.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig('/data/HeatMap/graphs/train_loss.png')

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.savefig('/data/HeatMap/graphs/all_metrics.png')

print("TRAINING GRAPHS ARE CREATED")















#
