
import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, Input





def inceptionv2(image_size=80):
    inp=Input(shape=(1,image_size,image_size))
    x = Conv2D(64, (7,7), activation='relu')(inp)
    #(74x74x64)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    #(37x37x64)
    x = Conv2D(128, (6,6), activation='relu')(x)
    x = BatchNormalization()(x)
    #(32x32x128)
    x = MaxPooling2D(pool_size=(2,2))(x)
    #(16x16x128)
    #Inception layers. Output here: (14x14x32)
    tower_1 = Conv2D(96, (1,1), padding='same', activation='relu')(x)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Conv2D(128, (3,3), padding='same', activation='relu')(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_2 = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_4 = BatchNormalization()(tower_4)
    tower_5 = keras.layers.concatenate([tower_1, tower_2, tower_3,tower_4], axis = 1)
    #Output here: (16x16x256)

    tower_6 = Conv2D(128, (1,1), padding='same', activation='relu')(tower_5)
    tower_6 = BatchNormalization()(tower_6)
    tower_6 = Conv2D(192, (3,3), padding='same', activation='relu')(tower_6)
    tower_6 = BatchNormalization()(tower_6)
    tower_7 = Conv2D(32, (1,1), padding='same', activation='relu')(tower_5)
    tower_7 = BatchNormalization()(tower_7)
    tower_7 = Conv2D(96, (5,5), padding='same', activation='relu')(tower_7)
    tower_7 = BatchNormalization()(tower_7)
    tower_8 = MaxPooling2D((3,3), strides=(1,1), padding='same')(tower_5)
    tower_8 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_8)
    tower_8 = BatchNormalization()(tower_8)
    tower_9 = Conv2D(128, (1,1), padding='same', activation='relu')(tower_5)
    tower_9 = BatchNormalization()(tower_9)
    tower_10 = keras.layers.concatenate([tower_6, tower_7, tower_8, tower_9], axis = 1)
    tower_11 = MaxPooling2D(pool_size=(2,2))(tower_10)
    #Output here: (8x8x480)

    tower_12 = Conv2D(96, (1,1), padding='same', activation='relu')(tower_11)
    tower_12 = BatchNormalization()(tower_12)
    tower_12 = Conv2D(208, (3,3), padding='same', activation='relu')(tower_12)
    tower_12 = BatchNormalization()(tower_12)
    tower_13 = Conv2D(16, (1,1), padding='same', activation='relu')(tower_11)
    tower_13 = BatchNormalization()(tower_13)
    tower_13 = Conv2D(48, (5,5), padding='same', activation='relu')(tower_13)
    tower_13 = BatchNormalization()(tower_13)
    tower_14 = MaxPooling2D((3,3), strides=(1,1), padding='same')(tower_11)
    tower_14 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_14)
    tower_14 = BatchNormalization()(tower_14)
    tower_15 = Conv2D(192, (1,1), padding='same', activation='relu')(tower_11)
    tower_15 = BatchNormalization()(tower_15)
    tower_16 = keras.layers.concatenate([tower_12, tower_12, tower_14, tower_15], axis = 1)
    #Output here: (8x8x512)




    tower_17 = Conv2D(112, (1,1), padding='same', activation='relu')(tower_16)
    tower_17 = Conv2D(224, (3,3), padding='same', activation='relu')(tower_17)
    tower_18 = Conv2D(24, (1,1), padding='same', activation='relu')(tower_16)
    tower_18 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_18)
    tower_19 = MaxPooling2D((3,3), strides=(1,1), padding='same')(tower_16)
    tower_19 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_19)
    tower_20 = Conv2D(160, (1,1), padding='same', activation='relu')(tower_16)
    tower_21 = keras.layers.concatenate([tower_17, tower_18, tower_19, tower_20], axis = 1)
    #Output here: (8x8x512)

    tower_22 = Conv2D(144, (1,1), padding='same', activation='relu')(tower_21)
    tower_22 = Conv2D(288, (3,3), padding='same', activation='relu')(tower_22)
    tower_23 = Conv2D(32, (1,1), padding='same', activation='relu')(tower_21)
    tower_23 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_23)
    tower_24 = MaxPooling2D((3,3), strides=(1,1), padding='same')(tower_21)
    tower_24 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_24)
    tower_25 = Conv2D(128, (1,1), padding='same', activation='relu')(tower_21)
    tower_26 = keras.layers.concatenate([tower_22, tower_23, tower_24, tower_25], axis = 1)
    pool = MaxPooling2D(pool_size=(2,2))(tower_26)
    #Output here: (4x4x512)


    tower_27 = Conv2D(144, (1,1), padding='same', activation='relu')(pool)
    tower_27 = Conv2D(288, (3,3), padding='same', activation='relu')(tower_27)
    tower_28 = Conv2D(32, (1,1), padding='same', activation='relu')(pool)
    tower_28 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_28)
    tower_29 = MaxPooling2D((3,3), strides=(1,1), padding='same')(pool)
    tower_29 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_29)
    tower_30 = Conv2D(112, (1,1), padding='same', activation='relu')(pool)
    tower_31 = keras.layers.concatenate([tower_27, tower_28, tower_29, tower_30], axis = 1)
    #Output here: (4x4x528)


    output_1 = AveragePooling2D(pool_size=(2,2))(tower_31)
    output_1 = Conv2D(256, (1,1), padding='same', activation='relu')(output_1)
    output_1=Flatten()(output_1)
    output_1=Dense(1024, activation='relu')(output_1)
    output_1=Dropout(0.5)(output_1)
    output_1=Dense(1024, activation='relu')(output_1)
    output_1=Dropout(0.5)(output_1)
    y1=Dense(5, activation='softmax',name='y1')(output_1)

    model = Model(inputs=inp, outputs=y1)
    return model


