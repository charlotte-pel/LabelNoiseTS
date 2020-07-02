from EvalAlgo import EvalFunc
import time
from pathlib import Path
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GRU, Bidirectional
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical


def tempCNNWork(path, nbClass, noiseArray, nbFirstRun, nbLastRun, seed, systematicChange=False, nbunits_conv=64, nbunits_fc=256):
    """
    TempCNN evaluation function
    :param path: Path to dataset
    :param nbClass: Number of class in classification
    :param noiseArray: Array containing all noise level
    :param nbFirstRun: First run number (1)
    :param nbLastRun: Last run number (10)
    :param seed: seed for shuffle data
    :param systematicChange: True if noise is systematic change, False if noise is random:
    :return: dfAccuracyTempCNN, dfAccuracyTempCNNCsv
    """

    path = Path(path)

    resultsArray = np.array([])
    indexRunList = []
    algoName = 'TempCNN'
    for i in range(nbFirstRun, nbLastRun + 1):
        print('TempCNN')
        print('Run ' + str(i))
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            (Xtrain, Xtest, ytrain, ytest) = EvalFunc.getXtrainXtestYtrainYtest(path, j, i, seed, systematicChange)

            accuracy_score = TempCNN(Xtrain, ytrain, Xtest, ytest, nbClass, nbunits_conv, nbunits_fc)

            results.append(accuracy_score)

        resultsArray = np.append(resultsArray, values=results, axis=0)

    (dfAccuracyTempCNN, dfAccuracyTempCNNCsv) = EvalFunc.makeDfAccuracyMeanStd(resultsArray, noiseArray, algoName,
                                                                               nbFirstRun,
                                                                               nbLastRun, indexRunList)

    return dfAccuracyTempCNN, dfAccuracyTempCNNCsv


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# ---------------------			Temp CNN			---------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Deep Learning for the classification of Sentinel-2 image time series
# Code come from: https://github.com/charlotte-pel/igarss2019-dl4sits
# Authors: Dr. Charlotte Pelletier, Professor Geoffrey I. Webb, Dr. Francois Petitjean
# -----------------------------------------------------------------------
def TempCNN(X_train, y_train, X_test, y_test, nbClass, nbunits_conv, nbunits_fc):
    classifier_type = 'TempCNN'
    # Parameters
    # -- general
    nchannels = 1
    # -- deep learning
    n_epochs = 35
    batch_size = 32
    val_rate = 0

    # Evaluated metrics
    eval_label = ['OA', 'train_loss', 'train_time', 'test_time']

    # # Output filenames
    # res_file = './resultOA-' + classifier_type + '.csv'
    res_mat = np.zeros((len(eval_label), 1))
    model_file = './model-' + classifier_type + '.h5'
    # conf_file = './confMatrix-' + classifier_type + '.csv'
    # acc_loss_file = './trainingHistory-' + classifier_type + '.csv'  # -- only for deep learning models

    # Training
    # ---- Pre-processing train data
    X_train = reshape_data(X_train, nchannels)
    min_per, max_per = computingMinMax(X_train)
    X_train = normalizingData(X_train, min_per, max_per)
    y_train_one_hot = to_categorical(y_train, nbClass)
    X_test = reshape_data(X_test, nchannels)
    X_test = normalizingData(X_test, min_per, max_per)
    y_test_one_hot = to_categorical(y_test, nbClass)

    model = Archi_TempCNN(X_train, nbClass, nbunits_conv, nbunits_fc)

    res_mat[0], res_mat[1], model, model_hist, res_mat[2], res_mat[3] = \
        trainTestModel(model, X_train, y_train_one_hot, X_test, y_test_one_hot, model_file, n_epochs=n_epochs,
                       batch_size=batch_size)
    # saveLossAcc(model_hist, acc_loss_file)
    p_test = model.predict(x=X_test)

    print('Overall accuracy (OA): ', res_mat[0])
    print('Train loss: ', res_mat[1])
    print('Training time (s): ', res_mat[2])
    print('Test time (s): ', res_mat[3])

    return float(res_mat[0])


def Archi_TempCNN(X, nbclasses, nbunits_conv, nbunits_fc):
    # -- get the input sizes
    m, L, depth = X.shape
    input_shape = (L, depth)

    # -- parameters of the architecture
    l2_rate = 1.e-6
    dropout_rate = 0.5
    nb_conv = 3
    nb_fc = 1
    # nbunits_conv = 64  # 32
    # nbunits_fc = 256  # 128

    # Define the input placeholder.
    X_input = Input(input_shape)

    # -- nb_conv CONV layers
    X = X_input
    for add in range(nb_conv):
        X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate),
                              dropout_rate=dropout_rate)
    # -- Flatten + 	1 FC layers
    X = Flatten()(X)
    for add in range(nb_fc):
        X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)

    # -- SOFTMAX layer
    out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))

    # Create model.
    return Model(inputs=X_input, outputs=out, name='Archi_3CONV64_1FC256')


def trainTestModel(model, X_train, Y_train_onehot, X_test, Y_test_onehot, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                          epsilon=None, decay=decay)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # ---- monitoring the minimum loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    start_train_time = time.time()
    hist = model.fit(x=X_train, y=Y_train_onehot, epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(X_test, Y_test_onehot),
                     verbose=1, callbacks=callback_list)
    train_time = round(time.time() - start_train_time, 2)

    # -- download the best model
    del model
    model = load_model(out_model_file)
    start_test_time = time.time()
    test_loss, test_acc = model.evaluate(x=X_test, y=Y_test_onehot,
                                         batch_size=128, verbose=0)
    test_time = round(time.time() - start_test_time, 2)

    return test_acc, np.min(hist.history['loss']), model, hist.history, train_time, test_time


def softmax(X, nbclasses, **params):
    kernel_regularizer = params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = params.setdefault("kernel_initializer", "glorot_uniform")
    return Dense(nbclasses, activation='softmax',
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(X)


def conv_bn(X, **conv_params):
    nbunits = conv_params["nbunits"];
    kernel_size = conv_params["kernel_size"];

    strides = conv_params.setdefault("strides", 1)
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    Z = Conv1D(nbunits, kernel_size=kernel_size,
               strides=strides, padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer)(X)

    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)


def conv_bn_relu_drop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = conv_bn_relu(X, **conv_params)
    return Dropout(dropout_rate)(A)


def conv_bn_relu(X, **conv_params):
    Znorm = conv_bn(X, **conv_params)
    return Activation('relu')(Znorm)


def fc_bn(X, **fc_params):
    nbunits = fc_params["nbunits"];

    kernel_regularizer = fc_params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = fc_params.setdefault("kernel_initializer", "he_normal")

    Z = Dense(nbunits, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(X)
    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)


def fc_bn_relu(X, **fc_params):
    Znorm = fc_bn(X, **fc_params)
    return Activation('relu')(Znorm)


def fc_bn_relu_drop(X, **fc_params):
    dropout_rate = fc_params.setdefault("dropout_rate", 0.5)
    A = fc_bn_relu(X, **fc_params)
    return Dropout(dropout_rate)(A)


def computingMinMax(X, per=2):
    min_per = np.percentile(X, per, axis=(0, 1))
    max_per = np.percentile(X, 100 - per, axis=(0, 1))
    return min_per, max_per


def normalizingData(X, min_per, max_per):
    return (X - min_per) / (max_per - min_per)


def reshape_data(X, nchannels):
    """
        Reshaping (feature format (3 bands): d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...)
        INPUT:
            -X: original feature vector ()
            -feature_strategy: used features (options: SB, NDVI, SB3feat)
            -nchannels: number of channels
        OUTPUT:
            -new_X: data in the good format for Keras models
    """

    return X.reshape(X.shape[0], int(X.shape[1] / nchannels), nchannels)
