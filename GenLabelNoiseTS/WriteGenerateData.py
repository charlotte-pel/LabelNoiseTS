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
        :param rep: name of the rep
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param csv: Csv Option True or False
        :return:
        """
        filename = rep+filename
        hdf = pd.HDFStore(filename)
        hdf.put('headerNoise', pd.DataFrame(None, columns=['name']))
        dfLutCsv = pd.DataFrame(None, columns=['dict', 'csv'])
        hdf.put('header', dfHeader)
        if csv is False:
            hdf.put('data', dfData)
            hdf.put('lut', dfLutCsv)
        else:
            dfData.to_csv(rep+'data.csv', index=False)
            dfLutCsv.to_csv(rep+'lut.csv',index=False)
            dfCsv = pd.DataFrame(np.array([rep+'data.csv']), columns=['csv'])
            hdf.put('csvFile', dfCsv)
        hdf.close()

    @staticmethod
    def writeGenerateNoisyData(filename,rep, noiseLevel, dfNoisy, systematicChange, csv=False, dict=None):
        """

        :param filename: the name of the file in which the data will be entered
        :param rep: name of the rep
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dfNoisy:
        :param systematicChange: Name of systematic schema class WheatToBarley_...
        :param csv: Csv Option True or False
        :param dict: Dict used for generate noise
        :return:
        """
        filename = rep + filename
        if systematicChange is not None:
            name = systematicChange
            if csv is False:
                dfLutCsv = pd.DataFrame(pd.read_hdf(filename,'lut'))
                dfLutCsv = dfLutCsv.append(
                    pd.DataFrame(np.array([[str(dict), rep + name + '.csv']]), columns=['dict', 'csv']))
                hdf = pd.HDFStore(filename)
                hdf.put(name, dfNoisy)
                hdf.put('lut',dfLutCsv)
                hdf.close()
                WriteGenerateData.writeHeaderNoise(filename, name)
            else:
                dfLutCsv = pd.DataFrame(pd.read_csv(rep+'lut.csv'))
                dfLutCsv = dfLutCsv.append(pd.DataFrame([[str(dict),rep + name + '.csv']], columns=['dict','csv']))
                dfLutCsv.to_csv(rep+'lut.csv', index=False)

                dfNoisy.to_csv(rep+name + '.csv', index=False)
                dfCsv = pd.read_hdf(filename, 'csvFile')
                dfCsv = pd.DataFrame(dfCsv)
                dfCsv = dfCsv.append(pd.DataFrame(np.array([rep+name + '.csv']), columns=['csv']))
                hdf = pd.HDFStore(filename)
                hdf.put('csvFile', dfCsv)
                hdf.close()
        else:
            name = 'random_' + str(int(noiseLevel * 100))
            if csv is False:
                dfLutCsv = pd.DataFrame(pd.read_hdf(filename, 'lut'))
                dfLutCsv = dfLutCsv.append(
                    pd.DataFrame(np.array([[str(dict), rep + name + '.csv']]), columns=['dict', 'csv']))
                hdf = pd.HDFStore(filename)
                hdf.put('random_' + str(int(noiseLevel * 100)), dfNoisy)
                hdf.put('lut', dfLutCsv)
                hdf.close()
                WriteGenerateData.writeHeaderNoise(filename, name)
            else:
                dfLutCsv = pd.DataFrame(pd.read_csv(rep + 'lut.csv'))
                dfLutCsv = dfLutCsv.append(pd.DataFrame([[str(dict), rep + name + '.csv']], columns=['dict', 'csv']))
                dfLutCsv.to_csv(rep + 'lut.csv', index=False)

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
    def writeTest(filename,rep,test ,csv=False):
        """

        :param filename: the name of the file in which the data will be entered
        :param rep: name of the rep
        :param test: Name of new Noisy dataset
        :param csv: Csv Option True or False
        :return:
        """
        filename = rep + filename
        if csv is False:
            hdf = pd.HDFStore(filename)
            hdf.put('test', test)
            hdf.close()
        else:
            test.to_csv(rep+'test.csv', index=False)
            dfCsv = pd.read_hdf(filename, 'csvFile')
            dfCsv = pd.DataFrame(dfCsv)
            dfCsv = dfCsv.append(pd.DataFrame(np.array([rep+'test.csv']), columns=['csv']))
            hdf = pd.HDFStore(filename)
            hdf.put('csvFile', dfCsv)
            hdf.close()

    @staticmethod
    def updateDfHeader(filename,rep, dfHeader):
        filename = rep + filename
        hdf = pd.HDFStore(filename)
        hdf.put('header', dfHeader)
        hdf.close()
