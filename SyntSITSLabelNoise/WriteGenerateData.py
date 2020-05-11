import pandas as pd
import numpy as np
import os
import warnings

class WriteGenerateData:

    warnings.filterwarnings("ignore")

    @staticmethod
    def writeGenerateDataToH5(filename, dfHeader, dfData):
        """

        :param filename: the name of the file in which the data will be entered
        :param dfHeader:
        :param dfData:
        :return:
        """
        if not os.path.isfile(filename):
            hdf = pd.HDFStore(filename)
            hdf.put('headerNoise', pd.DataFrame(None, columns=['name']))
            hdf.close()
        hdf = pd.HDFStore(filename)
        hdf.put('header', dfHeader)
        hdf.put('data', dfData)
        hdf.close()

    @staticmethod
    def writeGenerateNoisyData(filename, noiseLevel, dfNoisy, systematicChange):
        """

        :param filename: the name of the file in which the data will be entered
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dfNoisy:
        :param systematicChange: Name of systematic schema class WheatToBarley_...
        :return:
        """
        hdf = pd.HDFStore(filename)
        if systematicChange is not None:
            hdf.put('noiseData_'+systematicChange+ str(int(noiseLevel * 100)), dfNoisy)
            name = 'noiseData_'+systematicChange+ str(int(noiseLevel * 100))
        else:
            hdf.put('noiseData_'+str(int(noiseLevel*100)), dfNoisy)
            name = 'noiseData_'+str(int(noiseLevel*100))
        hdf.close()
        WriteGenerateData.writeHeaderNoise(filename, name)

    @staticmethod
    def writeHeaderNoise(filename,name):
        """

        :param filename: the name of the file in which the data will be entered
        :param name: Name of new Noisy dataset
        :return:
        """
        dfHeaderNoise = pd.read_hdf(filename, 'headerNoise')
        dfHeaderNoise = pd.DataFrame(dfHeaderNoise)
        dfHeaderNoise = dfHeaderNoise.append(pd.DataFrame(np.array([[name]]),columns=['name']))
        hdf = pd.HDFStore(filename)
        hdf.put('headerNoise', dfHeaderNoise)
        hdf.close()

    @staticmethod
    def writeTest(filename,test):
        """

        :param filename: the name of the file in which the data will be entered
        :param test: Name of new Noisy dataset
        :return:
        """
        hdf = pd.HDFStore(filename)
        hdf.put('test', test)
        hdf.close()