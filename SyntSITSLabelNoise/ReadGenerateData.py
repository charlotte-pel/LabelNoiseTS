import pandas as pd
import numpy as np
import warnings

class ReadGenerateData:

    warnings.filterwarnings("ignore")

    @staticmethod
    def readGenerateDataH5DataFrame(filename):
        """

        :param filename: the name of the file in which the data will be entered
        :return:
        """
        dfheader = pd.read_hdf(filename, 'header')
        dfData = pd.read_hdf(filename, 'data')
        dfheader = pd.DataFrame(dfheader)
        dfData = pd.DataFrame(dfData)
        nbClass = len(dfheader) - 1
        dates = dfheader[0][0]
        classNames = []
        for i in range(1, nbClass + 1):
            classNames.append(dfheader[0][i][0])
        samplesClass = dfData
        nbPixelClass = [len(samplesClass.loc[(samplesClass['label'] == i)]) for i in classNames]
        return nbClass, dates, classNames, samplesClass, nbPixelClass

    @staticmethod
    def getAlreadyGenNoise(filename,name):
        """

        :param filename: the name of the file in which the data will be entered
        :param name: Name of new Noisy dataset
        :return: True if and only if dataset named "name" is found
        """
        found = False
        dfName = pd.read_hdf(filename, 'headerNoise')
        dfName = pd.DataFrame(dfName)
        dfName = np.array(dfName)
        if name in dfName:
            found = True
        return found

    @staticmethod
    def getByNameNoise(filename,name):
        """

        :param filename: the name of the file in which the data will be entered
        :param name: Name of new Noisy dataset
        :return: DataFrame format like that columns=['pixid', 'noisy', 'label']
        """
        dfNoise = pd.read_hdf(filename, name)
        dfNoise = pd.DataFrame(dfNoise)
        return dfNoise
