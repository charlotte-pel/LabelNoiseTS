from GenLabelNoiseTS.GenLabelNoiseTS import *


def getXtrainXtestYtrainYtest(path, systematicChange, seed, noiseLevel, run):
    generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=path + 'Run' + str(run) + '/', csv=True,
                                verbose=False)
    if systematicChange is False:
        (Xtrain, ytrain) = generator.getNoiseDataXY(noiseLevel)
    elif systematicChange is True:
        a = {'Corn': 'Corn_Ensilage', 'Corn_Ensilage': 'Sorghum', 'Sorghum': 'Sunflower',
             'Sunflower': 'Soy',
             'Soy': 'Corn'}
        (Xtrain, ytrain) = generator.getNoiseDataXY(noiseLevel, a)

    randomState = np.random.RandomState(seed)
    Xtrain = randomState.permutation(Xtrain)
    randomState = np.random.RandomState(seed)
    ytrain = randomState.permutation(ytrain)

    (Xtest, ytest) = generator.getTestData(otherPath=path + '/Run10/')

    return Xtrain, Xtest, ytrain, ytest


def makeDfAccuracyMeanStd(resultsArray, noiseArray, algoName, nbFirstRun, nbLastRun, indexRunList):
    dfAccuracyCsv = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList)).reshape(nbLastRun, len(noiseArray)), columns=noiseArray).T

    dfAccuracyMean = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).mean()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStd = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStd.rename(columns={0: algoName + ' NDVI STD'}, inplace=True)
    dfAccuracyMean.rename(columns={0: algoName + ' NDVI'}, inplace=True)
    dfAccuracy = dfAccuracyMean.join(dfAccuracyStd)

    return dfAccuracy, dfAccuracyCsv
