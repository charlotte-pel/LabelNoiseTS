import numpy as np

from EvalAlgo import EvalFunc
from TempCNN.TempCNN import TempCNN


def tempCNNWork(path, nbClass, noiseArray, nbFirstRun, nbLastRun, seed, systematicChange=False):
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

            accuracy_score = TempCNN('TempCNN', Xtrain, ytrain, Xtest, ytest, nbClass)

            results.append(accuracy_score)

        resultsArray = np.append(resultsArray, values=results, axis=0)

    (dfAccuracyTempCNN, dfAccuracyTempCNNCsv) = EvalFunc.makeDfAccuracyMeanStd(resultsArray, noiseArray, algoName, nbFirstRun,
                                                                    nbLastRun, indexRunList)

    return dfAccuracyTempCNN, dfAccuracyTempCNNCsv
