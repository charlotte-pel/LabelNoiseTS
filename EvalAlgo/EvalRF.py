from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from EvalAlgo import EvalFunc
from GenLabelNoiseTS.GenLabelNoiseTS import *


def randomForestWork(path, noiseArray, noFirstRun, noLastRun, seed, NJOBS, systematicChange=False):
    """
    Random Forest evaluation function
    :param path: Path to dataset
    :param noiseArray: Array containing all noise level
    :param noFirstRun: First run number (1)
    :param noLastRun: Last run number (10)
    :param seed: seed for shuffle data
    :param NJOBS: Number of cores used
    :param systematicChange: True if noise is systematic change, False if noise is random
    :return: dfAccuracyRF, dfAccuracyRFCsv
    """

    path = Path(path)

    resultsArray = np.array([])
    indexRunList = []
    algoName = 'RF'

    for i in range(noFirstRun, noLastRun + 1):
        print('RF')
        print('Run ' + str(i))
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            (Xtrain, Xtest, ytrain, ytest) = EvalFunc.getXtrainXtestYtrainYtest(path, j, i, seed, systematicChange)

            clf = RandomForestClassifier(n_estimators=200, max_depth=25, max_features='sqrt', n_jobs=NJOBS)
            clf.fit(Xtrain, ytrain)
            ytest_pred = clf.predict(Xtest)
            results.append(accuracy_score(ytest, ytest_pred))
            C = confusion_matrix(ytest, ytest_pred)
            del clf

        resultsArray = np.append(resultsArray, values=results, axis=0)

    (dfAccuracyRF, dfAccuracyRFCsv) = EvalFunc.makeDfAccuracyMeanStd(resultsArray, noiseArray, algoName, noFirstRun,
                                                                     noLastRun, indexRunList)

    return dfAccuracyRF, dfAccuracyRFCsv
