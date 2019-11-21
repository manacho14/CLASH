
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:28:38 2017

@author: manacho
"""
import os, sys, json
import keras
import numpy as np
import pandas as pd
from model import inceptionv2
from keras import backend as K
import matplotlib.pyplot as plt
from preprocess import Preprocess
from keras import losses, optimizers, initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]="0"

plt.switch_backend('agg')
K.set_image_dim_ordering('th')


def get_params(dataset):
    with open('./{}/config.json'.format(dataset), 'r') as f:
        params = json.load(f)
    return params


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def generator(nb_batches_train, train_data, nb_random_perturbations, dataset):
    while True:
        for i in range(nb_batches_train):
            samples = np.random.choice(len(train_data), 128)
            train_data_epoch = train_data.iloc[samples]
            preprocess = Preprocess(train_data_epoch, '{}/stamps/'.format(dataset))
            X, y = preprocess.get_data()
            datagen = ImageDataGenerator(rotation_range=360, width_shift_range=0.05, height_shift_range=0.05,horizontal_flip=True, vertical_flip=True)
            data = datagen.flow(X, y, batch_size = X.shape[0], shuffle=False).next()
            yield data[0],data[1]


def delete_duplicates(df1, df2):
    indexes = []
    for i in range(len(df1)):
        for j in range(len(df2)):
            if df1['ID'][i].split('_')[0]+"_"+df1['ID'][i].split('_')[2] == df2['ID'][j].split('_')[0]+"_"+df2['ID'][j].split('_')[2]:
                indexes.append(i)
    df1 = df1.drop(indexes)

    return df1


def plot_learning_curve(loss, val_loss, score):
    plt.clf()
    plt.plot(loss, color='k')
    plt.plot(val_loss, color='b')
    plt.axhline(score, linestyle='--', color='r')
    plt.title('model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation','Test'], loc='upper right')
    plt.plot(len(loss)-1, score, '*', color='r', markersize=10)
    plt.savefig("lossvsepoch.png")



if __name__ == '__main__':

    dataset = str(sys.argv[1])

    data = pd.read_csv('{}/{}_labels.csv'.format(dataset,dataset), index_col = None)
    data = data.sample(frac=1).reset_index(drop=True)

    params = get_params(dataset)

    val_data = data[:params['val_size']].reset_index(drop = True)
    test_data = data[params['val_size']:params['val_size'] + params['test_size']].reset_index(drop = True)
    train_data = data[params['val_size'] + params['test_size']:].reset_index(drop = True)

    model = inceptionv2(params['image_size'])
    if dataset == 'CLASH':
        model.load_weights('weights-improvement-55-0.12.hdf5')
        test_data = delete_duplicates(test_data, train_data)
        val_data = delete_duplicates(val_data, train_data)

    model.compile(loss = root_mean_squared_error, optimizer = keras.optimizers.Adam(lr = params['learning_rate']))


    early_stopping = EarlyStopping(monitor = 'val_loss', patience = params['early_stopping'], mode = 'auto')

    nb_batches_train = int(train_data.shape[0]/params["batch_size_train"])


    prep_val = Preprocess(val_data, '{}/stamps/'.format(dataset))
    prep_test = Preprocess(test_data, '{}/stamps/'.format(dataset))

    history = model.fit_generator(generator(nb_batches_train, train_data, params['random_perturbations'], dataset),
                                  steps_per_epoch=nb_batches_train, nb_epoch = params['num_epochs'], validation_data= prep_val.get_data(),
                                  callbacks= [early_stopping])

    X_test,y_test = prep_test.get_data()
    score = model.evaluate(X_test,y_test)
    print('Score:', score)

    plot_learning_curve(history.history['loss'],history.history['val_loss'], score)

