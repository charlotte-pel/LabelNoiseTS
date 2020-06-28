from EvalAlgo.EvalRF import *
from EvalAlgo.EvalSVM import *
from EvalAlgo.EvalTempCNN import *


def EvalAlgo(path, nbClass, seed, systematicChange=False):
    NJOBS = 8
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 1

    (dfAccuracySVML, dfAccuracyCsvSVML) = svmWork(path, 'linear', noiseArray, nbFirstRun, nbLastRun, seed,
                                                  systematicChange)

    (dfAccuracySVMRBF, dfAccuracyCsvSVMRBF) = svmWork(path, 'rbf', noiseArray, nbFirstRun, nbLastRun, seed,
                                                      systematicChange)

    (dfAccuracyRF, dfAccuracyCsvRF) = randomForestWork(NJOBS, path, noiseArray, nbFirstRun, nbLastRun, seed,
                                                       systematicChange)

    (dfAccuracyTempCNN, dfAccuracyCsvTempCNN) = tempCNNWork(path, nbClass, noiseArray, nbFirstRun, nbLastRun, seed,
                                                            systematicChange)

    if systematicChange is False:
        dfAccuracyRF.to_csv(path + "AccuracyRF.csv")
        dfAccuracySVML.to_csv(path + "AccuracySVM_Linear.csv")
        dfAccuracySVMRBF.to_csv(path + "AccuracySVM_RBF.csv")
        dfAccuracyTempCNN.to_csv(path + "AccuracyTempCNN.csv")

        dfAccuracyCsvRF.to_csv(path + "AccuracyCsvRF.csv")
        dfAccuracyCsvSVML.to_csv(path + "AccuracyCsvSVM_Linear.csv")
        dfAccuracyCsvSVMRBF.to_csv(path + "AccuracyCsvSVM_RBF.csv")
        dfAccuracyCsvTempCNN.to_csv(path + "AccuracyCsvTempCNN.csv")

    elif systematicChange is True:
        dfAccuracyRF.to_csv(path + "AccuracyScRF.csv")
        dfAccuracySVML.to_csv(path + "AccuracyScSVM_Linear.csv")
        dfAccuracySVMRBF.to_csv(path + "AccuracyScSVM_RBF.csv")
        dfAccuracyTempCNN.to_csv(path + "AccuracyScTempCNN.csv")

        dfAccuracyCsvRF.to_csv(path + "AccuracyCsvScRF.csv")
        dfAccuracyCsvSVML.to_csv(path + "AccuracyCsvScSVM_Linear.csv")
        dfAccuracyCsvSVMRBF.to_csv(path + "AccuracyCsvScSVM_RBF.csv")
        dfAccuracyCsvTempCNN.to_csv(path + "AccuracyCsvScTempCNN.csv")


def visualisationEval(path, nbClass, systematicChange=False):
    if systematicChange is False:
        dfAccuracyRF = pd.read_csv(path + 'AccuracyRF.csv', index_col=0)
        dfAccuracySVML = pd.read_csv(path + 'AccuracySVM_Linear.csv', index_col=0)
        dfAccuracySVMRBF = pd.read_csv(path + 'AccuracySVM_RBF.csv', index_col=0)
        dfAccuracyTempCNN = pd.read_csv(path + 'AccuracyTempCNN.csv', index_col=0)

    elif systematicChange is True:
        dfAccuracyRF = pd.read_csv(path + 'AccuracyScRF.csv', index_col=0)
        dfAccuracySVML = pd.read_csv(path + 'AccuracyScSVM_Linear.csv', index_col=0)
        dfAccuracySVMRBF = pd.read_csv(path + 'AccuracyScSVM_RBF.csv', index_col=0)
        dfAccuracyTempCNN = pd.read_csv(path + 'AccuracyScTempCNN.csv', index_col=0)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    dfAccuracyRF.plot(y='RF NDVI', kind='line', legend=True, yerr='RF NDVI STD', ax=ax)
    dfAccuracySVML.plot(y='SVM-LINEAR NDVI', kind='line', legend=True, yerr='SVM-LINEAR NDVI STD', ax=ax)
    dfAccuracySVMRBF.plot(y='SVM-RBF NDVI', kind='line', legend=True, yerr='SVM-RBF NDVI STD', ax=ax)
    dfAccuracyTempCNN.plot(y='TempCNN NDVI', kind='line', legend=True, yerr='TempCNN NDVI STD', ax=ax)

    plt.title(nbClass)
    plt.xlabel('Niveau de bruit')
    plt.ylabel('Taux de bonnes classification')
    plt.grid()
    plt.axis([-0.01, 1.01, 0, 1])
    plt.show()
