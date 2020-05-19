import pandas as pd
import numpy as np
import warnings

class ReadGenerateData:

    warnings.filterwarnings("ignore")

    @staticmethod
    def readGenerateDataH5DataFrame(filename,rep,csv=False):
        """

        :param filename: the name of the file in which the data will be entered
        :param rep: name of the rep
        :param csv: Csv Option True or False
        :return:
        """
        filename = rep + filename
        dfheader = pd.read_hdf(filename, 'header')
        dfheader = pd.DataFrame(dfheader)
        nbClass = len(dfheader) - 1
        dates = dfheader[0][0]
        classNames = []
        for i in range(1, nbClass + 1):
            classNames.append(dfheader[0][i][0])
        if csv is False:
            dfData = pd.read_hdf(filename, 'data')
            dfData = pd.DataFrame(dfData)
        else:
            dfCsv = pd.read_hdf(filename, 'csvFile')
            npCsv = np.array(pd.DataFrame(dfCsv))[0]
            dfData = pd.read_csv(npCsv[0])
        samplesClass = dfData
        nbPixelClass = [len(samplesClass.loc[(samplesClass['label'] == i)]) for i in classNames]
        return nbClass, dates, classNames, samplesClass, nbPixelClass

    @staticmethod
    def getAlreadyGenNoise(filename,rep,name, csv=False):
        """

        :param filename: the name of the file in which the data will be entered
        :param rep: name of the rep
        :param name: Name of new Noisy dataset
        :param csv: Csv Option True or False
        :return: True if and only if dataset named "name" is found
        """
        filename = rep + filename
        found = False
        if csv is False:
            dfName = pd.read_hdf(filename, 'headerNoise')
            dfName = pd.DataFrame(dfName)
        else:
            dfCsv = pd.read_hdf(filename, 'csvFile')
            dfName = pd.DataFrame(dfCsv)
            dfName = np.array(dfName)
            name = [rep+name+'.csv']
        dfName = np.array(dfName)
        if name in dfName:
            found = True
        return found

    @staticmethod
    def getByNameNoise(filename,rep,name,csv=False):
        """

        :param filename: the name of the file in which the data will be entered
        :param rep: name of the rep
        :param name: Name of new Noisy dataset
        :param csv: Csv Option True or False
        :return: DataFrame format like that columns=['pixid', 'noisy', 'label']
        """
        filename = rep + filename
        if csv is False:
            dfNoise = pd.read_hdf(filename, name)
        else:
            dfNoise = pd.read_csv(rep+name+'.csv')
        dfNoise = pd.DataFrame(dfNoise)
        return dfNoise
