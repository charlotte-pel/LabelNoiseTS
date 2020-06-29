#!/usr/bin/python

"""
	Temp CNN
"""
import sys

from TempCNN.dl_func import *
from TempCNN.res_func import *
from TempCNN.sits_func import *


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# ---------------------			Temp CNN			---------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Deep Learning for the classification of Sentinel-2 image time series
# Code come from: https://github.com/charlotte-pel/igarss2019-dl4sits
# Authors: Dr. Charlotte Pelletier, Professor Geoffrey I. Webb, Dr. Francois Petitjean
# -----------------------------------------------------------------------
def TempCNN(classifier_type, X_train, y_train, X_test, y_test, nbClass):
    classif_type = ["RF", "TempCNN", "GRU-RNNbi", "GRU-RNN"]
    if classifier_type not in classif_type:
        print("ERR: select an available classifier (RF, TempCNN, GRU-RNNbi or GRU-RNN)")
        sys.exit(1)

    dl_flag = True
    if classifier_type == "RF":
        dl_flag = False

    # Parameters
    # -- general
    nchannels = 1
    # -- deep learning
    n_epochs = 35
    batch_size = 32
    val_rate = 0

    # Evaluated metrics
    eval_label = ['OA', 'train_loss', 'train_time', 'test_time']

    # Output filenames
    res_file = './resultOA-' + classifier_type + '.csv'
    res_mat = np.zeros((len(eval_label), 1))
    model_file = './model-' + classifier_type + '.h5'
    conf_file = './confMatrix-' + classifier_type + '.csv'
    acc_loss_file = './trainingHistory-' + classifier_type + '.csv'  # -- only for deep learning models

    # Training
    if dl_flag:  # -- deep learning approaches
        # ---- Pre-processing train data
        X_train = reshape_data(X_train, nchannels)
        min_per, max_per = computingMinMax(X_train)
        X_train = normalizingData(X_train, min_per, max_per)
        y_train_one_hot = to_categorical(y_train, nbClass)
        X_test = reshape_data(X_test, nchannels)
        X_test = normalizingData(X_test, min_per, max_per)
        y_test_one_hot = to_categorical(y_test, nbClass)

        if classifier_type == "TempCNN":
            model = Archi_TempCNN(X_train, nbClass)
        elif classifier_type == "GRU-RNNbi":
            model = Archi_GRURNNbi(X_train, nbClass)
        elif classifier_type == "GRU-RNN":
            model = Archi_GRURNN(X_train, nbClass)

        if val_rate == 0:
            res_mat[0], res_mat[1], model, model_hist, res_mat[2], res_mat[3] = \
                trainTestModel(model, X_train, y_train_one_hot, X_test, y_test_one_hot, model_file, n_epochs=n_epochs,
                               batch_size=batch_size)
        saveLossAcc(model_hist, acc_loss_file)
        p_test = model.predict(x=X_test)

        print('Overall accuracy (OA): ', res_mat[0])
        print('Train loss: ', res_mat[1])
        print('Training time (s): ', res_mat[2])
        print('Test time (s): ', res_mat[3])

        return float(res_mat[0])
