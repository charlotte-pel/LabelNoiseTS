import sys

sys.path.append('../')
from GenLabelNoiseTS.GeneratorData import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():
    # noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    # rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/data/'
    # pathTwoClass = rootPath + 'TwoClass/'
    # generator = GeneratorData(filename="dataFrame.h5", rep=pathTwoClass + 'Run' + str(1) + '/', csv=True,verbose=False)
    # (X, y) = generator.getDataXY()
    # (Xtraina, ytrainb) = generator.getNoiseDataXY(noiseArray[2])
    # (Xtrain, ytrain) = generator.getNoiseDataXY(noiseArray[10])
    # print(generator.getNoiseMatrix(y, ytrainb))
    # print(generator.getNoiseMatrix(y, ytrain))
    #
    # rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/file/'
    # pathTwoClass = rootPath + ''
    # generator = GeneratorData(filename="dataFrame.h5", rep=pathTwoClass, csv=True, verbose=False,
    #                           classList=('Corn', 'Corn_Ensilage'), pathInitFile='../init_param_file.csv')
    # (X, y) = generator.getDataXY()
    # (Xtraina, ytrainb) = generator.getNoiseDataXY(noiseArray[2])
    # (Xtrain, ytrain) = generator.getNoiseDataXY(noiseArray[10])
    # print(generator.getNoiseMatrix(y, ytrainb))
    # print(generator.getNoiseMatrix(y, ytrain))

    NJOBS = 4
    rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/data/'
    pathTwoClass = rootPath + 'TwoClass/'
    pathFiveClass = rootPath + 'FiveClass/'
    pathTenClass = rootPath + 'TenClass/'
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 10

    (dfAccuracyMeanSVM,dfAccuracyStdSVM) = svmWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun)
    (dfAccuracyMeanRF,dfAccuracyStdRF) = randomForestWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun)

    fig, ax = plt.subplots()
    dfAccuracyMeanSVM.plot(kind='line', legend=True, yerr=dfAccuracyStdSVM, ax=ax)
    dfAccuracyMeanRF.plot(kind='line', legend=True, yerr=dfAccuracyStdRF, ax=ax)

    plt.grid()
    plt.axis([0, 1, 0, 1])
    plt.show()


def svmWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun):
    resultsArray = np.array([])
    indexRunList = []
    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GeneratorData(filename="dataFrame.h5", rep=pathTwoClass + 'Run' + str(i) + '/', csv=True,
                                      verbose=False)
            (Xtrain, ytrain) = generator.getNoiseDataXY(j)
            (Xtest, ytest) = generator.getTestData(otherPath=pathTwoClass + '/Run10/')
            Xshape = Xtrain.shape
            meanXj = np.mean(Xtrain, axis=0)
            stdXj = np.std(Xtrain, axis=0)
            XTrainNorm = []
            XTestNorm = []
            for n, k in zip(Xtrain, Xtest):
                l = 0
                for o, m in zip(n, k):
                    XTrainNorm.append((o - meanXj[l]) / stdXj[l])
                    XTestNorm.append((m - meanXj[l]) / stdXj[l])
                    l += 1
            XTrainNorm = np.array(XTrainNorm).reshape(Xshape)
            XTestNorm = np.array(XTestNorm).reshape(Xshape)
            parameters = {'C': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4]}
            svc = svm.SVC(kernel='linear', decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(XTrainNorm, ytrain)
            val = parameters['C'][np.max(clf.cv_results_['rank_test_score']) - 1]
            parameters = {
                'C': [val * 2 ** -1, val * 2 ** -(4 / 5), val * 2 ** -(3 / 5), val * 2 ** -(2 / 5), val * 2 ** -(1 / 5),
                      val * 2 ** 0, val * 2 ** (1 / 5), val * 2 ** (2 / 5), val * 2 ** (3 / 5), val * 2 ** (4 / 5)]}
            del svc
            del clf
            svc = svm.SVC(kernel='linear', decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(Xtrain, ytrain)
            val2 = parameters['C'][np.max(clf.cv_results_['rank_test_score']) - 1]

            clf = svm.SVC(C=val2, kernel='linear', decision_function_shape='ovo', random_state=0)
            clf.fit(XTrainNorm, ytrain)
            ytest_pred = clf.predict(XTestNorm)
            results.append(accuracy_score(ytest, ytest_pred))
            del svc
            del clf
            del generator
        resultsArray = np.append(resultsArray, values=results, axis=0)

    dfAccuracyMeanSVM = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).mean()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdSVM = pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))),
                                 columns=noiseArray, index=indexRunList).std()
    dfAccuracyMeanSVM.rename(columns={0: 'SVM-Linear NDVI'}, inplace=True)

    return dfAccuracyMeanSVM,dfAccuracyStdSVM


def randomForestWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun):
    resultsArray = np.array([])
    indexRunList = []

    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GeneratorData(filename="dataFrame.h5", rep=pathTwoClass + 'Run' + str(i) + '/', csv=True,
                                      verbose=False)
            (Xtrain, ytrain) = generator.getNoiseDataXY(j)
            (Xtest, ytest) = generator.getTestData(otherPath=pathTwoClass + '/Run10/')
            clf = RandomForestClassifier(n_estimators=200, max_depth=25, max_features='sqrt', n_jobs=NJOBS,
                                         random_state=0)
            clf.fit(Xtrain, ytrain)
            ytest_pred = clf.predict(Xtest)
            results.append(accuracy_score(ytest, ytest_pred))
            C = confusion_matrix(ytest, ytest_pred)
            del clf
            del generator
        resultsArray = np.append(resultsArray, values=results, axis=0)

    dfAccuracyMeanRF = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).mean()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdRF = pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))),
                                 columns=noiseArray, index=indexRunList).std()
    dfAccuracyMeanRF.rename(columns={0: 'RF NDVI'}, inplace=True)

    return dfAccuracyMeanRF,dfAccuracyStdRF

if __name__ == "__main__": main()
