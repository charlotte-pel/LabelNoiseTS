import sys

from EvalAlgo.EvalRF import *
from EvalAlgo.EvalSVM import *
from EvalAlgo.EvalTempCNN import *


def EvalAlgo(path, nbClass, seed, systematicChange=False, outPathResults=None, nbunits_conv=64, nbunits_fc=256):
    """
    Functions for evaluation of RandomForest, SVM-Linear, SVM-RBF, TempCNN
    :param path: Path to dataset
    :param nbClass: Number of class in classification
    :param seed: seed for shuffle data
    :param systematicChange: True if noise is systematic change, False if noise is random
    :param outPathResults: path to results directory
    :return: None
    """

    if path is None:
        if nbClass == 2:
            path = './results/Evals/TwoClass/'
            nbClass = 'Two class'
        elif nbClass == 5:
            if systematicChange is False:
                path = './results/Evals/FiveClass/'
                nbClass = 'Five class'
            else:
                path = './results/Evals/FiveClass/systematicChange/'
                nbClass = 'Five class Systematic Change'
        elif nbClass == 10:
            path = './results/Evals/TenClass/'
            nbClass = 'Ten class'

    NJOBS = 8
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 10

    (dfAccuracySVML, dfAccuracyCsvSVML) = svmWork(path, 'linear', noiseArray, nbFirstRun, nbLastRun, seed,
                                                  systematicChange)

    (dfAccuracySVMRBF, dfAccuracyCsvSVMRBF) = svmWork(path, 'rbf', noiseArray, nbFirstRun, nbLastRun, seed,
                                                      systematicChange)

    (dfAccuracyRF, dfAccuracyCsvRF) = randomForestWork(path, noiseArray, nbFirstRun, nbLastRun, seed, NJOBS,
                                                       systematicChange)

    (dfAccuracyTempCNN, dfAccuracyCsvTempCNN) = tempCNNWork(path, nbClass, noiseArray, nbFirstRun, nbLastRun, seed,
                                                            systematicChange, nbunits_conv=nbunits_conv,
                                                            nbunits_fc=nbunits_fc)

    if systematicChange is False:
        dfAccuracyRF.to_csv(outPathResults + "AccuracyRF.csv")
        dfAccuracySVML.to_csv(outPathResults + "AccuracySVM_Linear.csv")
        dfAccuracySVMRBF.to_csv(outPathResults + "AccuracySVM_RBF.csv")
        dfAccuracyTempCNN.to_csv(outPathResults + "AccuracyTempCNN.csv")

        dfAccuracyCsvRF.to_csv(outPathResults + "AccuracyCsvRF.csv")
        dfAccuracyCsvSVML.to_csv(outPathResults + "AccuracyCsvSVM_Linear.csv")
        dfAccuracyCsvSVMRBF.to_csv(outPathResults + "AccuracyCsvSVM_RBF.csv")
        dfAccuracyCsvTempCNN.to_csv(outPathResults + "AccuracyCsvTempCNN.csv")

    elif systematicChange is True:
        dfAccuracyRF.to_csv(outPathResults + "AccuracyScRF.csv")
        dfAccuracySVML.to_csv(outPathResults + "AccuracyScSVM_Linear.csv")
        dfAccuracySVMRBF.to_csv(outPathResults + "AccuracyScSVM_RBF.csv")
        dfAccuracyTempCNN.to_csv(outPathResults + "AccuracyScTempCNN.csv")

        dfAccuracyCsvRF.to_csv(outPathResults + "AccuracyCsvScRF.csv")
        dfAccuracyCsvSVML.to_csv(outPathResults + "AccuracyCsvScSVM_Linear.csv")
        dfAccuracyCsvSVMRBF.to_csv(outPathResults + "AccuracyCsvScSVM_RBF.csv")
        dfAccuracyCsvTempCNN.to_csv(outPathResults + "AccuracyCsvScTempCNN.csv")


def visualisationEval(nbClass, path=None, systematicChange=False):
    """
    Evaluation visualisation function
    :param path: Path to dataset
    :param nbClass: Number of class in classification
    :param systematicChange: True if noise is systematic change, False if noise is random
    :return: Show plot result of evaluation
    """

    if path is None:
        if nbClass == 2:
            path = './results/Evals/TwoClass/'
            nbClass = 'Two class'
        elif nbClass == 5:
            if systematicChange is False:
                path = './results/Evals/FiveClass/'
                nbClass = 'Five class'
            else:
                path = './results/Evals/FiveClass/systematicChange/'
                nbClass = 'Five class Systematic Change'
        elif nbClass == 10:
            path = './results/Evals/TenClass/'
            nbClass = 'Ten class'

    if systematicChange is False or None:
        if systematicChange is None:
            print('Error systematicChange is not False or True !!!')
            print('systematicChange will be set to False')
        dfAccuracyRF = pd.read_csv(path + 'AccuracyRF.csv', index_col=0)
        dfAccuracySVML = pd.read_csv(path + 'AccuracySVM_Linear.csv', index_col=0)
        dfAccuracySVMRBF = pd.read_csv(path + 'AccuracySVM_RBF.csv', index_col=0)
        dfAccuracyTempCNN = pd.read_csv(path + 'AccuracyTempCNN.csv', index_col=0)

    elif systematicChange is True:
        dfAccuracyRF = pd.read_csv(path + 'AccuracyScRF.csv', index_col=0)
        dfAccuracySVML = pd.read_csv(path + 'AccuracyScSVM_Linear.csv', index_col=0)
        dfAccuracySVMRBF = pd.read_csv(path + 'AccuracyScSVM_RBF.csv', index_col=0)
        dfAccuracyTempCNN = pd.read_csv(path + 'AccuracyScTempCNN.csv', index_col=0)
    else:
        print('Error systematicChange is not False/None or True!!!')
        print('End of program')
        sys.exit(0)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    dfAccuracyRF.plot(y='RF NDVI', kind='line', legend=True, yerr='RF NDVI STD', ax=ax)
    dfAccuracySVML.plot(y='SVM-Linear NDVI', kind='line', legend=True, yerr='SVM-Linear NDVI STD', ax=ax)
    dfAccuracySVMRBF.plot(y='SVM-RBF NDVI', kind='line', legend=True, yerr='SVM-RBF NDVI STD', ax=ax)
    dfAccuracyTempCNN.plot(y='TempCNN NDVI', kind='line', legend=True, yerr='TempCNN NDVI STD', ax=ax)

    plt.title(nbClass)
    plt.xlabel('Noise Level')
    plt.ylabel('Overall Accuracy')
    plt.grid()
    plt.axis([-0.01, 1.01, 0, 1])
    plt.show()


# def visuTempCNN(nbClass, path=None, systematicChange=False):
#     if path is None:
#         if nbClass == 2:
#             path = './results/Evals/TwoClass/TempCNN/'
#             nbClass = 'Two class'
#         elif nbClass == 5:
#             path = './results/Evals/FiveClass/TempCNN/'
#             nbClass = 'Five class'
#         elif nbClass == 10:
#             path = './results/Evals/TenClass/TempCNN/'
#             nbClass = 'Ten class'
#
#     if systematicChange is False or None:
#         if systematicChange is None:
#             print('Error systematicChange is not False or True !!!')
#             print('systematicChange will be set to False')
#
#         dfAccuracyTempCNN1 = pd.read_csv(path + 'AccuracyTempCNN1.csv', index_col=0)
#         dfAccuracyTempCNN2 = pd.read_csv(path + 'AccuracyTempCNN2.csv', index_col=0)
#         dfAccuracyTempCNN3 = pd.read_csv(path + 'AccuracyTempCNN3.csv', index_col=0)
#
#     elif systematicChange is True:
#         dfAccuracyTempCNN1 = pd.read_csv(path + 'AccuracyScTempCNN1.csv', index_col=0)
#         dfAccuracyTempCNN2 = pd.read_csv(path + 'AccuracyScTempCNN2.csv', index_col=0)
#         dfAccuracyTempCNN3 = pd.read_csv(path + 'AccuracyScTempCNN3.csv', index_col=0)
#     else:
#         print('Error systematicChange is not False/None or True!!!')
#         print('End of program')
#         sys.exit(0)
#
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal', 'box')
#     dfAccuracyTempCNN1.plot(y='TempCNN NDVI Arch 1', kind='line', legend=True, yerr='TempCNN NDVI STD', ax=ax)
#     dfAccuracyTempCNN2.plot(y='TempCNN NDVI Arch 2', kind='line', legend=True, yerr='TempCNN NDVI STD', ax=ax)
#     dfAccuracyTempCNN3.plot(y='TempCNN NDVI Arch 3', kind='line', legend=True, yerr='TempCNN NDVI STD', ax=ax)
#
#     plt.title(nbClass)
#     plt.xlabel('Noise Level')
#     plt.ylabel('Overall Accuracy')
#     plt.grid()
#     plt.axis([-0.01, 1.01, 0, 1])
#     plt.show()
