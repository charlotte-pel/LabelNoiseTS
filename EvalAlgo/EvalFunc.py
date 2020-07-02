from GenLabelNoiseTS.GenLabelNoiseTS import *
from pathlib import Path

def getXtrainXtestYtrainYtest(path, noiseLevel, run, seed, systematicChange):
    """
    Function to get Xtrain, Xtest, Ytrain, Ytest
    :param path: Path to dataset
    :param noiseLevel: Noise level for Xtrain data
    :param run: run number (1 -> 10)
    :param seed: seed for shuffle data
    :param systematicChange: True if noise is systematic change, False if noise is random
    :return: Xtrain, Xtest, ytrain, ytest
    """
    path = Path(path)
    generator = GenLabelNoiseTS(filename="dataFrame.h5", dir=path / ('Run' + str(run) + '/'), csv=True,
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

    (Xtest, ytest) = generator.getTestData(otherPath=path / '/Run10/')

    return Xtrain, Xtest, ytrain, ytest


def makeDfAccuracyMeanStd(resultsArray, noiseArray, algoName, nbFirstRun, nbLastRun, indexRunList):
    """
    Function to make Accuracy Dataframe (Mean and STD)
    :param resultsArray: results of evaluation
    :param noiseArray: Array containing all noise level
    :param algoName: Algorithm name like RF, SVM-Linear, ...
    :param nbFirstRun: First run number (1)
    :param nbLastRun: Last run number (10)
    :param indexRunList: List of run
    :return: dfAccuracy, dfAccuracyCsv
    """
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
