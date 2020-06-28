from TempCNN.TempCNN import TempCNN
from GenLabelNoiseTS.GenLabelNoiseTS import *


def tempCNNWork(path, nbClass, noiseArray, nbFirstRun, nbLastRun, seed, systematicChange=False):
    resultsArray = np.array([])
    indexRunList = []

    for i in range(nbFirstRun, nbLastRun + 1):
        print('TempCNN')
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

            accuracy_score = TempCNN('TempCNN', Xtrain, ytrain, Xtest, ytest, nbClass)

            results.append(accuracy_score)

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
    dfAccuracyStdRF.rename(columns={0: 'TempCNN NDVI STD'}, inplace=True)
    dfAccuracyMeanRF.rename(columns={0: 'TempCNN NDVI'}, inplace=True)
    dfAccuracyRF = dfAccuracyMeanRF.join(dfAccuracyStdRF)

    return dfAccuracyRF, dfAccuracyCsv
