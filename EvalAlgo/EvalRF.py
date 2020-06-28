from EvalAlgo import EvalFunc
from GenLabelNoiseTS.GenLabelNoiseTS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def randomForestWork(NJOBS, path, noiseArray, nbFirstRun, nbLastRun,seed, systematicChange=False):
    resultsArray = np.array([])
    indexRunList = []
    algoName = 'RF'
    for i in range(nbFirstRun, nbLastRun + 1):
        print('RF')
        print('Run ' + str(i))
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            (Xtrain, Xtest, ytrain, ytest) = EvalFunc.getXtrainXtestYtrainYtest(path, systematicChange, seed, j, i)

            clf = RandomForestClassifier(n_estimators=200, max_depth=25, max_features='sqrt', n_jobs=NJOBS)
            clf.fit(Xtrain, ytrain)
            ytest_pred = clf.predict(Xtest)
            results.append(accuracy_score(ytest, ytest_pred))
            C = confusion_matrix(ytest, ytest_pred)
            del clf

        resultsArray = np.append(resultsArray, values=results, axis=0)

    (dfAccuracyRF, dfAccuracyRFCsv) = EvalFunc.makeDfAccuracyMeanStd(resultsArray, noiseArray, algoName, nbFirstRun,
                                                                       nbLastRun, indexRunList)

    return dfAccuracyRF, dfAccuracyRFCsv
