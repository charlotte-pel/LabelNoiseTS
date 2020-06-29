from multiprocessing import Pool
from EvalAlgo.EvalAlgo import *


# from CNNTemp.train_classifier import *


def main():
    # rootPath = 'E:/Documents/Images/Desktop/StageIrisa/data/'
    rootPath = 'C:/Users/walkz/OneDrive/Bureau/StageIrisa/data/'

    pathTwoClass = rootPath + 'TwoClass/'
    pathFiveClass = rootPath + 'FiveClass/'
    pathTenClass = rootPath + 'TenClass/'

    systematicChange = False
    nbclass = 2
    seed = 0

    EvalAlgo(pathTwoClass, nbclass, seed, systematicChange)
    visualisationEval(pathTwoClass, 'Two classes', systematicChange)


def getNoiseMatrixForSpeRun(numRun, path):
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    generator = GenLabelNoiseTS(filename="dataFrame.h5", rep=path + 'Run' + str(numRun) + '/', csv=True,
                                verbose=False)
    (Xtrue, ytrue) = generator.getDataXY()
    for i in noiseArray:
        print('Run ' + str(numRun) + ' pour ' + str(i * 100) + '%')
        (Xtrain, ytrain) = generator.getNoiseDataXY(i)
        print(generator.getNoiseMatrix(ytrue, ytrain))


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