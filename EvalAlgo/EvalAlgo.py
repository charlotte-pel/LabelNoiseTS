from EvalAlgo.EvalRF import *
from EvalAlgo.EvalSVM import *


def EvalAlgo():
    NJOBS = 8
    rootPath = 'E:/Documents/Images/Desktop/StageIrisa/data/'
    pathTwoClass = rootPath + 'TwoClass/'
    pathFiveClass = rootPath + 'FiveClass/'
    pathTenClass = rootPath + 'TenClass/'
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 10

    (dfAccuracySVML,dfAccuracyCsvSVML) = svmWork(pathTwoClass, 'linear', noiseArray, nbFirstRun, nbLastRun)

    (dfAccuracySVMRBF,dfAccuracyCsvSVMRBF) = svmWork(pathTwoClass, 'rbf', noiseArray, nbFirstRun, nbLastRun)

    (dfAccuracyRF,dfAccuracyCsvRF) = randomForestWork(NJOBS, pathTwoClass, noiseArray, nbFirstRun, nbLastRun)

    dfAccuracyRF.to_csv(pathTwoClass + "AccuracyRF.csv")
    dfAccuracySVML.to_csv(pathTwoClass + "AccuracySVM_Linear.csv")
    dfAccuracySVMRBF.to_csv(pathTwoClass + "AccuracySVM_RBF.csv")

    dfAccuracyCsvRF.to_csv(pathTwoClass + "AccuracyCsvRF.csv")
    dfAccuracyCsvSVML.to_csv(pathTwoClass + "AccuracyCsvSVM_Linear.csv")
    dfAccuracyCsvSVMRBF.to_csv(pathTwoClass + "AccuracyCsvSVM_RBF.csv")

    fig, ax = plt.subplots()
    dfAccuracyRF.plot(y='RF NDVI', kind='line', legend=True, yerr='RF NDVI STD', ax=ax)
    dfAccuracySVML.plot(y='SVM-LINEAR NDVI', kind='line', legend=True, yerr='SVM-LINEAR NDVI STD', ax=ax)
    dfAccuracySVMRBF.plot(y='SVM-RBF NDVI', kind='line', legend=True, yerr='SVM-RBF NDVI STD', ax=ax)

    plt.grid()
    plt.axis([-0.01, 1.01, 0, 1])
    plt.show()
