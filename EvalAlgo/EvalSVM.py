from GenLabelNoiseTS.GenLabelNoiseTS import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def svmWork(path, kernel, noiseArray, nbFirstRun, nbLastRun, seed, systematicChange=False):
    resultsArray = np.array([])
    indexRunList = []
    for i in range(nbFirstRun, nbLastRun + 1):
        print(kernel)
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
                parameters = {
                    'C': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4],
                    'gamma': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4]}

            elif kernel == 'linear':
                parameters = {
                    'C': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4]}

            svc = svm.SVC(kernel=kernel)
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy', n_jobs=-1)
            clf.fit(XTrainNorm, ytrain.ravel())

            # dfResults = pd.DataFrame(clf.cv_results_)
            # dfResults.to_csv(path + 'dfResults' + kernel + '.csv')

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

            svc = svm.SVC(kernel=kernel)
            clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=None, scoring='accuracy', n_jobs=-1)
            clf.fit(Xtrain, ytrain.ravel())

            # dfResults = pd.DataFrame(clf.cv_results_)
            # dfResults.to_csv(path + 'dfResults2' + kernel + '.csv')

            valC = clf.best_params_['C']
            if kernel == 'rbf':
                valG = clf.best_params_['gamma']
                clf = svm.SVC(C=valC, gamma=valG, kernel='rbf')
            elif kernel == 'linear':
                clf = svm.SVC(C=valC, kernel='linear')
            clf.fit(XTrainNorm, ytrain)
            ytest_pred = clf.predict(XTestNorm)
            results.append(accuracy_score(ytest, ytest_pred))
            del svc
            del clf
            del generator
        resultsArray = np.append(resultsArray, values=results, axis=0)

    dfAccuracyCsv = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList)).reshape(nbLastRun, len(noiseArray)), columns=noiseArray).T

    dfAccuracyMeanSVM = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).mean()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdSVM = pd.DataFrame(np.array(
        pd.DataFrame(resultsArray.reshape(((nbLastRun - nbFirstRun + 1), len(noiseArray))), columns=noiseArray,
                     index=indexRunList).std()).reshape(1, len(noiseArray)), columns=noiseArray).T
    dfAccuracyStdSVM.rename(columns={0: 'SVM-' + kernel.upper() + ' NDVI STD'}, inplace=True)
    dfAccuracyMeanSVM.rename(columns={0: 'SVM-' + kernel.upper() + ' NDVI'}, inplace=True)
    dfAccuracySVM = dfAccuracyMeanSVM.join(dfAccuracyStdSVM)

    return dfAccuracySVM, dfAccuracyCsv
