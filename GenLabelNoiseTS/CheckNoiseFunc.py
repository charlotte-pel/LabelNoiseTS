from GenLabelNoiseTS.GenLabelNoiseTS import *
from multiprocessing import Pool


def getNoiseMatrixForSpeRun(numRun, path):
    """
    Get noise matrix for specific run
    :param numRun: Run number
    :param path: path to data
    :return: None -> Print noise matrix for each noise level
    """
    path = Path(path)
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    generator = GenLabelNoiseTS(filename="dataFrame.h5", dir=path / ('Run' + str(numRun) + '/'), csv=True,
                                verbose=False)
    (Xtrue, ytrue) = generator.getDataXY()
    for i in noiseArray:
        print('Run ' + str(numRun) + ' pour ' + str(i * 100) + '%')
        (Xtrain, ytrain) = generator.getNoiseDataXY(i)
        print(generator.getNoiseMatrix(ytrue, ytrain))


def checkNoiseForTwoFiveTenClass(verbose=False):
    """

    :param verbose: Verbose mode -> True or False
    :return: None -> Get error if noise generation is wrong
    """
    print('Check Starting....')
    with Pool(6) as p:
        p.starmap(checkGeneratingNoise,
                  [(500, 2, './data/TwoClass/', verbose),
                   (500, 5, './data/FiveClass/', verbose),
                   (500, 10, './data/TenClass/', verbose)])
    print('Check ending !!!')


def checkGeneratingNoise(noSamples, noClass, path, verbose=False):
    """

    :param noSamples: Number of sample per class
    :param noClass: Number of class
    :param path: path to data
    :param verbose: Verbose mode -> True or False
    :return: None -> Get error if noise generation is wrong
    """
    path = Path(path)
    noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
    noFirstRun = 1
    noLastRun = 10
    indexRunList = []
    noSamplesAllClass = noClass * noSamples
    for i in range(noFirstRun, noLastRun + 1):
        results = []
        indexRunList.append('Run' + str(i))
        for j in noiseArray:

            if verbose is True:
                print(
                    '------------------------------------------------------------------------------------------------')
                print('noClass = ' + str(noClass))
                print('NoiseLevel = ' + str(j))

            generator = GenLabelNoiseTS(filename="dataFrame.h5", dir=path / ('Run' + str(i) + '/'), csv=True,
                                        verbose=False)
            (Xtrain, ytrain) = generator.getNoiseDataXY(j)
            (Xtrue, ytrue) = generator.getDataXY()

            if verbose is True:
                print(
                    True if (np.sum(np.diag(generator.getNoiseMatrix(ytrue, ytrain))) == round(
                        noSamplesAllClass * (1 - j))) else False)

            assert (np.sum(np.diag(generator.getNoiseMatrix(ytrue, ytrain))) == round(
                noSamplesAllClass * (1 - j))), "\nError_Generating_Noise\n" + "noClass =" + str(
                noClass) + "Run = " + str(
                i) + "\n" + "Noise Level = " + str(j)

            if verbose is True:
                print(
                    '------------------------------------------------------------------------------------------------')
