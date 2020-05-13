import pandas as pd
import numpy as np
import os
import warnings


class WriteGenerateData:
    warnings.filterwarnings("ignore")

    @staticmethod
    def writeGenerateDataToH5(filename,rep, dfHeader, dfData, csv=False):
        """

        :param filename: the name of the file in which the data will be entered
        :param dfHeader:
        :param dfData:
        :return:
        """
        filename = rep+filename
        hdf = pd.HDFStore(filename)
        hdf.put('headerNoise', pd.DataFrame(None, columns=['name']))
        hdf.put('header', dfHeader)
        if csv is False:
            hdf.put('data', dfData)
        else:
            dfData.to_csv(rep+'data.csv', index=False)
            dfCsv = pd.DataFrame(np.array([rep+'data.csv']), columns=['csv'])
            hdf.put('csvFile', dfCsv)
        hdf.close()

    @staticmethod
    def writeGenerateNoisyData(filename,rep, noiseLevel, dfNoisy, systematicChange, csv=False):
        """

        :param filename: the name of the file in which the data will be entered
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dfNoisy:
        :param systematicChange: Name of systematic schema class WheatToBarley_...
        :return:
        """
        filename = rep + filename
        if systematicChange is not None:
            name = 'noiseData_' + systematicChange + str(int(noiseLevel * 100))
            if csv is False:
                hdf = pd.HDFStore(filename)
                hdf.put(name, dfNoisy)
                hdf.close()
                WriteGenerateData.writeHeaderNoise(filename, name)
            else:
                dfNoisy.to_csv(rep+name + '.csv', index=False)
                dfCsv = pd.read_hdf(filename, 'csvFile')
                dfCsv = pd.DataFrame(dfCsv)
                dfCsv = dfCsv.append(pd.DataFrame(np.array([rep+name + '.csv']), columns=['csv']))
                hdf = pd.HDFStore(filename)
                hdf.put('csvFile', dfCsv)
                hdf.close()
        else:
            name = 'noiseData_' + str(int(noiseLevel * 100))
            if csv is False:
                hdf = pd.HDFStore(filename)
                hdf.put('noiseData_' + str(int(noiseLevel * 100)), dfNoisy)
                hdf.close()
                WriteGenerateData.writeHeaderNoise(filename, name)
            else:
                dfNoisy.to_csv(rep+name + '.csv', index=False)
                dfCsv = pd.read_hdf(filename, 'csvFile')
                dfCsv = pd.DataFrame(dfCsv)
                dfCsv = dfCsv.append(pd.DataFrame(np.array([rep+name + '.csv']), columns=['csv']))
                hdf = pd.HDFStore(filename)
                hdf.put('csvFile', dfCsv)
                hdf.close()


    @staticmethod
    def writeHeaderNoise(filename, name):
        """

        :param filename: the name of the file in which the data will be entered
        :param name: Name of new Noisy dataset
        :return:
        """
        dfHeaderNoise = pd.read_hdf(filename, 'headerNoise')
        dfHeaderNoise = pd.DataFrame(dfHeaderNoise)
        dfHeaderNoise = dfHeaderNoise.append(pd.DataFrame(np.array([[name]]), columns=['name']))
        hdf = pd.HDFStore(filename)
        hdf.put('headerNoise', dfHeaderNoise)
        hdf.close()

    @staticmethod
    def writeTest(filename, test):
        """

        :param filename: the name of the file in which the data will be entered
        :param test: Name of new Noisy dataset
        :return:
        """
        hdf = pd.HDFStore(filename)
        hdf.put('test', test)
        hdf.close()
