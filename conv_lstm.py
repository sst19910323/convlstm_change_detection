from keras.layers import Convolution2D, ConvLSTM2D, MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D
from keras.layers import Conv2D,BatchNormalization,Reshape,Permute,Activation,Input
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import core, merge
from keras.layers.merge import concatenate
from keras.models import Model
from PIL import Image
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint

import batch_loader
import load_dataset


# image_path1 = r"D:\change_detection_dataset\time1\0_0_0.png"
# image_path2 = r"D:\change_detection_dataset\time2\0_0_0.png"
# label_path = r"D:\change_detection_dataset\label\0_0_0.png"
#
# image1 = np.array(Image.open(image_path1))
# image2 = np.array(Image.open(image_path2))
# label = np.array(Image.open(label_path))
#
# lstm_input = np.array([image1,image2])
# # lstm_input = np.append(image1,image2)
# lstm_input = np.expand_dims(lstm_input, 0)
#
# label = np.expand_dims(label, 2)
# label = np.expand_dims(label, 0)

img_w = 256
img_h = 256
channels = 3
time = 2
nClasses = 1


def Unet(nClasses, optimizer=None, input_width=256, input_height=256, nChannels=3):
    inputs = Input((nChannels, input_height, input_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)

    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)

    conv6 = Convolution2D(nClasses, 1, 1, activation="sigmoid")(conv5)
    # conv6 = Convolution2D(n_label, (1, 1), activation="softmax")(conv5)

    # conv6 = Convolution2D(nClasses, 1, 1, activation='sigmoid', border_mode='same')(conv5)
    # conv6 = core.Reshape((nClasses, input_height * input_width))(conv6)
    # conv6 = core.Permute((2, 1))(conv6)
    #
    # conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv6)
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

def lstm_unet_small(nClasses, optimizer=None, input_width=256, input_height=256, nChannels=3, nTime = 2):
    # inputs = Input((nChannels, input_height, input_width))
    inputs = Input((nTime, input_height, input_width, nChannels))
    conv_lstm1 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', input_shape=(nTime, input_width, input_height, nChannels), data_format='channels_last', return_sequences=True, dropout=0.2 ,recurrent_dropout = 0.2)(inputs)
    conv_lstm1 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True)(conv_lstm1)

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv_lstm1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)

    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)

    conv6 = Convolution2D(nClasses, 1, 1, activation="sigmoid")(conv5)
    # conv6 = Convolution2D(n_label, (1, 1), activation="softmax")(conv5)

    # conv6 = Convolution2D(nClasses, 1, 1, activation='sigmoid', border_mode='same')(conv5)
    # conv6 = core.Reshape((nClasses, input_height * input_width))(conv6)
    # conv6 = core.Permute((2, 1))(conv6)
    #
    # conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv6)
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

def lunet_m_small(nClasses = 2, optimizer=None, input_width=256, input_height=256, nChannels=3, nTime = 2):
    inputs = Input((nTime, input_height, input_width, nChannels))

    conv_lstm1 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', input_shape=(nTime, input_width, input_height, nChannels), data_format='channels_last', return_sequences=True, dropout=0.2 ,recurrent_dropout = 0.2)(inputs)
    conv_lstm1 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True)(conv_lstm1)

    conv1 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last',return_sequences=False)(conv_lstm1)
    pooling1 = MaxPooling3D(pool_size=(1, 2, 2))(conv_lstm1)

    conv_lstm2 = ConvLSTM2D(64, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dropout=0.2 ,recurrent_dropout = 0.2)(pooling1)
    conv_lstm2 = ConvLSTM2D(64, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=False)(conv_lstm2)

    # conv2 = ConvLSTM2D(64, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last',return_sequences=False)(conv_lstm2)
    pooling2 = MaxPooling2D(pool_size=(2, 2))(conv_lstm2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pooling2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv_lstm2], axis=3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)

    conv6 = Convolution2D(nClasses, 1, 1, activation="sigmoid")(conv5)
    # conv6 = Convolution2D(nClasses, (1, 1), activation="softmax")(conv5)

    # conv6 = Convolution2D(nClasses, 1, 1, activation='relu', border_mode='same')(conv5)
    # conv6 = core.Reshape((nClasses, input_height * input_width))(conv6)
    # conv6 = core.Permute((2, 1))(conv6)
    #
    # conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv6)

    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

def lunet_small(nClasses = nClasses, optimizer='Adam', input_width=256, input_height=256, nChannels=3, nTime = 2):
    inputs = Input((nTime, input_height, input_width, nChannels))

    conv_lstm1 = ConvLSTM2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same', input_shape=(nTime, input_width, input_height, nChannels), data_format='channels_last', return_sequences=True,dilation_rate = (2,2), dropout=0.2 ,recurrent_dropout = 0.2)(inputs)
    conv_lstm1 = ConvLSTM2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2))(conv_lstm1)
    Norm1 = BatchNormalization()(conv_lstm1)
    pooling1 = MaxPooling3D(pool_size=(1, 2, 2))(Norm1)

    conv_lstm2 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True,dilation_rate = (2,2), dropout=0.2 ,recurrent_dropout = 0.2)(pooling1)
    conv_lstm2 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2))(conv_lstm2)
    Norm2 = BatchNormalization()(conv_lstm2)
    pooling2 = MaxPooling3D(pool_size=(1, 2, 2))(Norm2)

    conv_lstm3 = ConvLSTM2D(64, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2), dropout=0.2 ,recurrent_dropout = 0.2)(pooling2)
    conv_lstm3 = ConvLSTM2D(64, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2))(conv_lstm3)
    Norm3 = BatchNormalization()(conv_lstm3)

    up1 = concatenate([UpSampling3D(size=(1, 2, 2))(Norm3), Norm2], axis=4)
    conv_lstm4 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2), dropout=0.2 ,recurrent_dropout = 0.2)(up1)
    conv_lstm4 = ConvLSTM2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2))(conv_lstm4)
    Norm4 = BatchNormalization()(conv_lstm4)

    up2 = concatenate([UpSampling3D(size=(1, 2, 2))(Norm4), conv_lstm1], axis=4)
    conv_lstm5 = ConvLSTM2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=True, dilation_rate = (2,2), dropout=0.2 ,recurrent_dropout = 0.2)(up2)
    conv_lstm5 = ConvLSTM2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last', return_sequences=False, dilation_rate = (2,2))(conv_lstm5)
    Norm5 = BatchNormalization()(conv_lstm5)

    conv6 = Convolution2D(nClasses, 1, 1, activation="sigmoid")(Norm5)

    # conv6 = Convolution2D(nClasses, (1, 1), activation="softmax")(conv5)

    # conv6 = Convolution2D(nClasses, 1, 1, activation='relu', border_mode='same')(conv5)
    # conv6 = core.Reshape((nClasses, input_height * input_width))(conv6)
    # conv6 = core.Permute((2, 1))(conv6)
    #
    # conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv6)
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
              metrics=['accuracy'])

    return model

model = lunet_small()

model.summary()

# Train_Images,Train_Labels,Test_Images,Test_Labels = load_dataset.load_Images()
# Train_Images,Train_Labels = load_dataset.load_Images()
# print("load_datasets")

train_generator = batch_loader.generate_batch(1,4,'D:/change_detection_dataset/time1','D:/change_detection_dataset/time2','D:/change_detection_dataset/label')

checkpointer = ModelCheckpoint(filepath="checkpoint_{epoch:02d}_acc_{acc:.2f}.hdf5",save_best_only=False, verbose=1,  period=2)

print("load_datasets")
model.fit_generator(generator=train_generator,epochs = 20, steps_per_epoch=10, verbose=1, callbacks=[checkpointer],validation_data=None)

# score = model.evaluate(Train_Images, Train_Labels, batch_size=8)

# fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)


# y = model.predict(lstm_input)
# print(y.shape)


