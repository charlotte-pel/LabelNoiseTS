import sys
sys.path.append('../')
from multiprocessing import Pool
from GenLabelNoiseTS.GenLabelNoiseTS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():

    NJOBS = 4
    rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/data/'
    pathTwoClass = rootPath + 'TwoClass/'
    pathFiveClass = rootPath + 'FiveClass/'
    pathTenClass = rootPath + 'TenClass/'
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 10

    # checkNoiseForTwoFiveTenClass()

    # (dfAccuracySVMRBF) = svmWork2(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray,
    #                               nbFirstRun, nbLastRun)
    # (dfAccuracySVML) = svmWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray,
    #                            nbFirstRun, nbLastRun)

    (dfAccuracySVML) = svmWork3(pathTwoClass, 'linear', noiseArray, nbFirstRun, nbLastRun)

    (dfAccuracyRF) = randomForestWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray,
                                      nbFirstRun, nbLastRun)

    # dfAccuracyRF.to_csv(pathTwoClass + "AccuracyRF.csv")
    # dfAccuracySVML.to_csv(pathTwoClass + "AccuracySVM_Linear.csv")
    # dfAccuracySVMRBF.to_csv(pathTwoClass + "AccuracySVM_RBF.csv")

    fig, ax = plt.subplots()
    dfAccuracyRF.plot(y='RF NDVI', kind='line', legend=True, yerr='RF NDVI STD', ax=ax)
    dfAccuracySVML.plot(y='SVM-LINEAR NDVI', kind='line', legend=True, yerr='SVM-LINEAR NDVI STD', ax=ax)
    #dfAccuracySVMRBF.plot(y='SVM-RBF NDVI', kind='line', legend=True, yerr='SVM-RBF NDVI STD', ax=ax)

    plt.grid()
    plt.axis([-0.01, 1.01, 0, 1])
    plt.show()

def svmWork3(path, kernel, noiseArray, nbFirstRun, nbLastRun):

    resultsArray = np.array([])
    indexRunList = []
    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=path + 'Run' + str(i) + '/', csv=True,
                                      verbose=False)
            (Xtrain, ytrain) = generator.getNoiseDataXY(j)
            (Xtest, ytest) = generator.getTestData(otherPath=path + '/Run10/')
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

            if kernel == 'rbf':
                valG = 1
                parameters = {
                    'C': [2 ** -1, 2 ** -(4 / 5), 2 ** -(3 / 5), 2 ** -(2 / 5),
                        2 ** -(1 / 5),
                        2 ** 0, 2 ** (1 / 5), 2 ** (2 / 5), 2 ** (3 / 5),
                        2 ** (4 / 5)],
                    'gamma': [2 ** -1, 2 ** -(4 / 5), 2 ** -(3 / 5), 2 ** -(2 / 5),
                              2 ** -(1 / 5),
                              2 ** 0, 2 ** (1 / 5), 2 ** (2 / 5), 2 ** (3 / 5),
                              2 ** (4 / 5)]}

            elif kernel == 'linear':
                parameters = {
                    'C': [2 ** -1, 2 ** -(4 / 5), 2 ** -(3 / 5), 2 ** -(2 / 5),
                        2 ** -(1 / 5),
                        2 ** 0, 2 ** (1 / 5), 2 ** (2 / 5), 2 ** (3 / 5),
                        2 ** (4 / 5)]}

            svc = svm.SVC(kernel=kernel, decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(XTrainNorm, ytrain)

            valC = clf.best_params_['C']
            if kernel == 'rbf':
                valG = clf.best_params_['gamma']

            del svc
            del clf

            if kernel == 'rbf':
                parameters = {
                    'C': [valC * 2 ** -1, valC * 2 ** -(4 / 5), valC * 2 ** -(3 / 5), valC * 2 ** -(2 / 5),
                          valC * 2 ** -(1 / 5),
                          valC * 2 ** 0, valC * 2 ** (1 / 5), valC * 2 ** (2 / 5), valC * 2 ** (3 / 5),
                          valC * 2 ** (4 / 5)],
                    'gamma': [valG * 2 ** -1, valG * 2 ** -(4 / 5), valG * 2 ** -(3 / 5), valG * 2 ** -(2 / 5),
                              valG * 2 ** -(1 / 5),
                              valG * 2 ** 0, valG * 2 ** (1 / 5), valG * 2 ** (2 / 5), valG * 2 ** (3 / 5),
                              valG * 2 ** (4 / 5)]}

            elif kernel == 'linear':
                parameters = {
                    'C': [valC * 2 ** -1, valC * 2 ** -(4 / 5), valC * 2 ** -(3 / 5), valC * 2 ** -(2 / 5),
                          valC * 2 ** -(1 / 5),
                          valC * 2 ** 0, valC * 2 ** (1 / 5), valC * 2 ** (2 / 5), valC * 2 ** (3 / 5),
                          valC * 2 ** (4 / 5)]}

            svc = svm.SVC(kernel=kernel, decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(Xtrain, ytrain)

            valC = clf.best_params_['C']
            if kernel == 'rbf':
                valG = clf.best_params_['gamma']
                clf = svm.SVC(C=valC, gamma=valG, kernel='rbf', decision_function_shape='ovo', random_state=0)
            elif kernel == 'linear':
                clf = svm.SVC(C=valC, kernel='linear', decision_function_shape='ovo', random_state=0)
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
    dfAccuracyStdSVM = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdSVM.rename(columns={0: 'SVM-'+kernel.upper()+' NDVI STD'}, inplace=True)
    dfAccuracyMeanSVM.rename(columns={0: 'SVM-'+kernel.upper()+' NDVI'}, inplace=True)
    dfAccuracySVM = dfAccuracyMeanSVM.join(dfAccuracyStdSVM)

    return dfAccuracySVM

def svmWork2(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun):
    resultsArray = np.array([])
    indexRunList = []
    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=pathTwoClass + 'Run' + str(i) + '/', csv=True,
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
            parameters = {'C': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4],
                          'gamma': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3,
                                    2 ** 4]}
            svc = svm.SVC(kernel='rbf', decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(XTrainNorm, ytrain)
            valC = clf.best_params_['C']
            valG = clf.best_params_['gamma']
            parameters = {
                'C': [valC * 2 ** -1, valC * 2 ** -(4 / 5), valC * 2 ** -(3 / 5), valC * 2 ** -(2 / 5),
                      valC * 2 ** -(1 / 5),
                      valC * 2 ** 0, valC * 2 ** (1 / 5), valC * 2 ** (2 / 5), valC * 2 ** (3 / 5),
                      valC * 2 ** (4 / 5)],
                'gamma': [valG * 2 ** -1, valG * 2 ** -(4 / 5), valG * 2 ** -(3 / 5), valG * 2 ** -(2 / 5),
                          valG * 2 ** -(1 / 5),
                          valG * 2 ** 0, valG * 2 ** (1 / 5), valG * 2 ** (2 / 5), valG * 2 ** (3 / 5),
                          valG * 2 ** (4 / 5)]}
            del svc
            del clf
            svc = svm.SVC(kernel='rbf', decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(Xtrain, ytrain)
            valC = clf.best_params_['C']
            valG = clf.best_params_['gamma']
            clf = svm.SVC(C=valC, gamma=valG, kernel='rbf', decision_function_shape='ovo', random_state=0)
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
    dfAccuracyStdSVM = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdSVM.rename(columns={0: 'SVM-RBF NDVI STD'}, inplace=True)
    dfAccuracyMeanSVM.rename(columns={0: 'SVM-RBF NDVI'}, inplace=True)
    dfAccuracySVM = dfAccuracyMeanSVM.join(dfAccuracyStdSVM)

    return dfAccuracySVM


def svmWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun):
    resultsArray = np.array([])
    indexRunList = []
    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=pathTwoClass + 'Run' + str(i) + '/', csv=True,
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
            valC = clf.best_params_['C']
            parameters = {
                'C': [valC * 2 ** -1, valC * 2 ** -(4 / 5), valC * 2 ** -(3 / 5), valC * 2 ** -(2 / 5),
                      valC * 2 ** -(1 / 5),
                      valC * 2 ** 0, valC * 2 ** (1 / 5), valC * 2 ** (2 / 5), valC * 2 ** (3 / 5),
                      valC * 2 ** (4 / 5)]}
            del svc
            del clf
            svc = svm.SVC(kernel='linear', decision_function_shape='ovo')
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy')
            clf.fit(Xtrain, ytrain)
            valC = clf.best_params_['C']

            clf = svm.SVC(C=valC, kernel='linear', decision_function_shape='ovo', random_state=0)
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
    dfAccuracyStdSVM = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdSVM.rename(columns={0: 'SVM-Linear NDVI STD'}, inplace=True)
    dfAccuracyMeanSVM.rename(columns={0: 'SVM-Linear NDVI'}, inplace=True)
    dfAccuracySVM = dfAccuracyMeanSVM.join(dfAccuracyStdSVM)

    return dfAccuracySVM


def randomForestWork(NJOBS, pathTwoClass, pathFiveClass, pathTenClass, noiseArray, nbFirstRun, nbLastRun):
    resultsArray = np.array([])
    indexRunList = []

    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:
            generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=pathTwoClass + 'Run' + str(i) + '/', csv=True,
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
    dfAccuracyStdRF = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdRF.rename(columns={0: 'RF NDVI STD'}, inplace=True)
    dfAccuracyMeanRF.rename(columns={0: 'RF NDVI'}, inplace=True)
    dfAccuracyRF = dfAccuracyMeanRF.join(dfAccuracyStdRF)

    return dfAccuracyRF


def checkNoiseForTwoFiveTenClass(verbose=False):
    print('Check Starting....')
    with Pool(6) as p:
        p.starmap(checkGeneratingNoise,
                  [(500, 2, 'TwoClass/', verbose),
                   (500, 5, 'FiveClass/', verbose),
                   (500, 10, 'TenClass/', verbose)])
    print('Check ending !!!')


def checkGeneratingNoise(nbSamples, nbClass, pathClass, verbose=False):
    rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/data/'
    pathClass = rootPath + pathClass
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    nbFirstRun = 1
    nbLastRun = 10
    indexRunList = []
    nbSamplesAllClass = nbClass * nbSamples
    for i in range(nbFirstRun, nbLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:

            if verbose is True:
                print(
                    '------------------------------------------------------------------------------------------------')
                print('nbClass = ' + str(nbClass))
                print('NoiseLevel = ' + str(j))

            generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=pathClass + 'Run' + str(i) + '/', csv=True,
                                      verbose=False)
            (Xtrain, ytrain) = generator.getNoiseDataXY(j)
            (Xtrue, ytrue) = generator.getDataXY()

            if verbose is True:
                print(
                    True if (np.sum(np.diag(generator.getNoiseMatrix(ytrue, ytrain))) == round(
                        nbSamplesAllClass * (1 - j))) else False)

            assert (np.sum(np.diag(generator.getNoiseMatrix(ytrue, ytrain))) == round(
                nbSamplesAllClass * (1 - j))), "\nError_Generating_Noise\n" + "nbClass =" + str(
                nbClass) + "Run = " + str(
                i) + "\n" + "Noise Level = " + str(j)

            if verbose is True:
                print(
                    '------------------------------------------------------------------------------------------------')


if __name__ == "__main__": main()

# Debug code for GenerateNoise -> First case without bug, second with bug

# noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
# rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/file'
# file = rootPath + '/'
# file2 = rootPath + '2/'
# generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=file, csv=True, verbose=False, pathInitFile='../init_param_file.csv',seedData=31521)
# generator2 = GenLabelNoiseTS(filename="dataFrame.h5", rep=file2, csv=True, verbose=False,
#                           pathInitFile='../init_param_file.csv',seedData=31521)
# (X, y) = generator.getDataXY()
# (X2, y2) = generator2.getDataXY()
# print('----------------------------------------------')
# print(generator.getMatrixClassInt())
# print(generator2.getMatrixClassInt())
# print('----------------------------------------------')
# a = {'Wheat':'Barley'}
# (Xtraina, ytrainb) = generator.getNoiseDataXY(0.95,a)
# print(np.diag(generator.getNoiseMatrix(y, ytrainb)))
# a = {'Wheat': ('Barley','Soy','Corn')}
# (Xtraina, ytrainb) = generator.getNoiseDataXY(0.95,a)
# print(np.diag(generator.getNoiseMatrix(y, ytrainb)))
# print('ERROR')
# a = {'Wheat': ('Barley','Soy','Build')}
# (Xtrain2, ytrain2) = generator2.getNoiseDataXY(0.1, a,seedNoise=73055)
# print(generator2.getNoiseMatrix(y2, ytrain2))
# print(np.diag(generator2.getNoiseMatrix(y2, ytrain2)))
# print('----------------------------------------------')
# print('GOOD')
# (Xtrain, ytrain) = generator.getNoiseDataXY(0.1, a, seedNoise=26675)
# print(generator.getNoiseMatrix(y, ytrain))
# print(np.diag(generator.getNoiseMatrix(y, ytrain)))
# (Xtraina, ytrainb) = generator.getNoiseDataXY(0.95)
# print(np.diag(generator.getNoiseMatrix(y, ytrainb)))
# (Xtrain, ytrain) = generator.getNoiseDataXY(noiseArray[10],a)
# {'dataSeed': 3428, "0.95_{'Wheat': ('Barley', 'Soy'), 'Barley': 'Wheat'}": 73055} -> Error
# {'dataSeed': 31521, "0.95_{'Wheat': ('Barley', 'Soy'), 'Barley': 'Wheat'}": 52036} -> Good
# print(generator.getSeedList())


# Other Test code
# -----------------------------------------------------------------------------------------------------------------
# noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
# rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/file'
# file = rootPath + '/'
# file2 = rootPath + '2/'
# generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=file, csv=True, verbose=False, pathInitFile='../init_param_file.csv',classList=('Corn', 'Corn_Ensilage'))
# (X, y) = generator.getDataXY()
# print('----------------------------------------------')
# #print(generator.getMatrixClassInt())
# print('----------------------------------------------')
# a = {'Wheat':'Barley'}
# a = None
# (Xtraina, ytrainb) = generator.getNoiseDataXY(0.95,a)
# #print(np.diag(generator.getNoiseMatrix(y, ytrainb)))
# print(generator.getDfHeader())
