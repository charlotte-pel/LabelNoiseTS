import sys

from EvalAlgo.EvalRF import *
from EvalAlgo.EvalSVM import *
from EvalAlgo.EvalTempCNN import *


def EvalAlgo(path, noClass, seed, systematicChange=False, outPathResults=None, nounits_conv=64, nounits_fc=256):
    """
    Functions for evaluation of RandomForest, SVM-Linear, SVM-RBF, TempCNN
    :param path: Path to dataset
    :param noClass: Number of class in classification
    :param seed: seed for shuffle data
    :param systematicChange: True if noise is systematic change, False if noise is random
    :param outPathResults: path to results directory
    :return: None
    """
    NJOBS = 8
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    noFirstRun = 1
    noLastRun = 10

    if outPathResults is None:
        if noClass == 2:
            outPathResults = Path('./results/evals/TwoClass/')
        elif noClass == 5:
            if systematicChange is False:
                outPathResults = Path('./results/evals/FiveClass/random/')
            else:
                outPathResults = Path('./results/evals/FiveClass/systematic/')
                noClass = 'Five class Systematic Change'
        elif noClass == 10:
            outPathResults = Path('./results/evals/TenClass/')
    else:
        outPathResults = Path(outPathResults)

    path = Path(path)

    (dfAccuracySVML, dfAccuracyCsvSVML) = svmWork(path, 'linear', noiseArray, noFirstRun, noLastRun, seed,
                                                  systematicChange)

    (dfAccuracySVMRBF, dfAccuracyCsvSVMRBF) = svmWork(path, 'rbf', noiseArray, noFirstRun, noLastRun, seed,
                                                      systematicChange)

    (dfAccuracyRF, dfAccuracyCsvRF) = randomForestWork(path, noiseArray, noFirstRun, noLastRun, seed, NJOBS,
                                                       systematicChange)

    (dfAccuracyTempCNN, dfAccuracyCsvTempCNN) = tempCNNWork(path, noClass, noiseArray, noFirstRun, noLastRun, seed,
                                                            systematicChange, nounits_conv=nounits_conv,
                                                            nounits_fc=nounits_fc)

    dfAccuracyRF.to_csv(outPathResults / "meanOA_RF.csv")
    dfAccuracySVML.to_csv(outPathResults / "meanOA_SVM_Linear.csv")
    dfAccuracySVMRBF.to_csv(outPathResults / "meanOA_SVM_RBF.csv")
    dfAccuracyTempCNN.to_csv(outPathResults / "meanOA_TempCNN.csv")

    dfAccuracyCsvRF.to_csv(outPathResults / "runOA_RF.csv")
    dfAccuracyCsvSVML.to_csv(outPathResults / "runOA_SVM_Linear.csv")
    dfAccuracyCsvSVMRBF.to_csv(outPathResults / "runOA_SVM_RBF.csv")
    dfAccuracyCsvTempCNN.to_csv(outPathResults / "runOA_TempCNN.csv")


def visualisationEval(path):
    """
    Evaluation visualisation function
    :param path: Path to dataset
    :return: Show plot result of evaluation
    """

    path = Path(path)
    if path.stem == 'random':
        noClass = path.parent.stem
    elif path.stem == 'systematic':
        noClass = path.parent.stem + ' Systematic change'
    else:
        noClass = path.stem

    dfAccuracyRF = pd.read_csv(path / 'meanOA_RF.csv', index_col=0)
    dfAccuracySVML = pd.read_csv(path / 'meanOA_SVM_Linear.csv', index_col=0)
    dfAccuracySVMRBF = pd.read_csv(path / 'meanOA_SVM_RBF.csv', index_col=0)
    dfAccuracyTempCNN = pd.read_csv(path / 'meanOA_TempCNN.csv', index_col=0)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    dfAccuracyRF.plot(y='RF NDVI', kind='line', legend=True, yerr='RF NDVI STD', ax=ax)
    dfAccuracySVML.plot(y='SVM-Linear NDVI', kind='line', legend=True, yerr='SVM-Linear NDVI STD', ax=ax)
    dfAccuracySVMRBF.plot(y='SVM-RBF NDVI', kind='line', legend=True, yerr='SVM-RBF NDVI STD', ax=ax)
    dfAccuracyTempCNN.plot(y='TempCNN NDVI', kind='line', legend=True, yerr='TempCNN NDVI STD', ax=ax)

    plt.title(noClass)
    plt.xlabel('Noise Level')
    plt.ylabel('Overall Accuracy')
    plt.grid()
    plt.axis([-0.01, 1.01, 0, 1])
    plt.show()
