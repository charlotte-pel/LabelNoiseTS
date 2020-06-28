from GenLabelNoiseTS.GenLabelNoiseTS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def randomForestWork(NJOBS, path, noiseArray, nbFirstRun, nbLastRun,seed, systematicChange=False):
    resultsArray = np.array([])
    indexRunList = []

    for i in range(nbFirstRun, nbLastRun + 1):
        print('RF')
        print('Run ' + str(i))
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=path + 'Run' + str(i) + '/', csv=True,
                                        verbose=False)
            if systematicChange is False:
                (Xtrain, ytrain) = generator.getNoiseDataXY(j)
            elif systematicChange is True:
                a = {'Corn': 'Corn_Ensilage', 'Corn_Ensilage': 'Sorghum', 'Sorghum': 'Sunflower',
                     'Sunflower': 'Soy',
                     'Soy': 'Corn'}
                (Xtrain, ytrain) = generator.getNoiseDataXY(j, a)

            randomState = np.random.RandomState(seed)
            Xtrain = randomState.permutation(Xtrain)
            randomState = np.random.RandomState(seed)
            ytrain = randomState.permutation(ytrain)

            (Xtest, ytest) = generator.getTestData(otherPath=path + '/Run10/')
            clf = RandomForestClassifier(n_estimators=200, max_depth=25, max_features='sqrt', n_jobs=NJOBS)
            clf.fit(Xtrain, ytrain)
            ytest_pred = clf.predict(Xtest)
            results.append(accuracy_score(ytest, ytest_pred))
            C = confusion_matrix(ytest, ytest_pred)
            del clf
            del generator
        resultsArray = np.append(resultsArray, values=results, axis=0)

    dfAccuracyCsv = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList)).reshape(nbLastRun, len(noiseArray)), columns=noiseArray).T

    dfAccuracyMeanRF = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).mean()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdRF = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdRF.rename(columns={0: 'RF NDVI STD'}, inplace=True)
    dfAccuracyMeanRF.rename(columns={0: 'RF NDVI'}, inplace=True)
    dfAccuracyRF = dfAccuracyMeanRF.join(dfAccuracyStdRF)

    return dfAccuracyRF, dfAccuracyCsv
