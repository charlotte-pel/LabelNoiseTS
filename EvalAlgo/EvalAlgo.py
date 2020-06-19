from EvalAlgo.EvalRF import *
from EvalAlgo.EvalSVM import *


def EvalAlgo(path):
    NJOBS = 8
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 10

    (dfAccuracySVML, dfAccuracyCsvSVML) = svmWork(path, 'linear', noiseArray, nbFirstRun, nbLastRun)

    (dfAccuracySVMRBF, dfAccuracyCsvSVMRBF) = svmWork(path, 'rbf', noiseArray, nbFirstRun, nbLastRun)

    (dfAccuracyRF, dfAccuracyCsvRF) = randomForestWork(NJOBS, path, noiseArray, nbFirstRun, nbLastRun)

    dfAccuracyRF.to_csv(path + "AccuracyRF.csv")
    dfAccuracySVML.to_csv(path + "AccuracySVM_Linear.csv")
    dfAccuracySVMRBF.to_csv(path + "AccuracySVM_RBF.csv")

    dfAccuracyCsvRF.to_csv(path + "AccuracyCsvRF.csv")
    dfAccuracyCsvSVML.to_csv(path + "AccuracyCsvSVM_Linear.csv")
    dfAccuracyCsvSVMRBF.to_csv(path + "AccuracyCsvSVM_RBF.csv")


def visualisationEval(path, nbClass):
    dfAccuracyRF = pd.read_csv(path + 'AccuracyRF.csv', index_col=0)
    dfAccuracySVML = pd.read_csv(path + 'AccuracySVM_Linear.csv', index_col=0)
    dfAccuracySVMRBF = pd.read_csv(path + 'AccuracySVM_RBF.csv', index_col=0)

    fig, ax = plt.subplots()
    dfAccuracyRF.plot(y='RF NDVI', kind='line', legend=True, yerr='RF NDVI STD', ax=ax)
    dfAccuracySVML.plot(y='SVM-LINEAR NDVI', kind='line', legend=True, yerr='SVM-LINEAR NDVI STD', ax=ax)
    dfAccuracySVMRBF.plot(y='SVM-RBF NDVI', kind='line', legend=True, yerr='SVM-RBF NDVI STD', ax=ax)

    plt.title(nbClass)
    plt.grid()
    plt.axis([-0.01, 1.01, 0, 1])
    plt.show()
