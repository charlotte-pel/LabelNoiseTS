import h5py

from SyntSITSLabelNoise.GenerateData import *
from SyntSITSLabelNoise.GeneratorNoise import *
from SyntSITSLabelNoise.WriteGenerateData import *
import numpy as np


class GeneratorData:

    def __init__(self, filename, seedData=0, csv=False, verbose=False, rep=''):
        """

        :param filename: Name of the file h5
        """
        self._verbose = verbose
        if self._verbose is True:
            print("Init Start !")
        self._rep = rep
        self._filename = filename
        self._dfTest = None
        self._csv = csv
        self._seedData = seedData
        # Test if not file exist
        if not os.path.isfile(self._rep + self._filename):
            (dfHeader, dfData) = self._genData()
            self._dfHeader = dfHeader
            self._dfData = dfData
            WriteGenerateData.writeGenerateDataToH5(self._filename, self._rep, self._dfHeader, self._dfData, self._csv)
            if self._verbose is True:
                print("Generate Data Done !")
        else:
            # File exist
            self._dfHeader = pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'header'))
            # Test if convert is needed
            if csv is False:
                try:
                    dfCsv = pd.read_hdf(self._rep + self._filename, 'csvFile')
                    npCsv = np.array(pd.DataFrame(dfCsv))
                    npCsv = npCsv.reshape((len(npCsv),))
                    self._dfData = pd.DataFrame(pd.read_csv(npCsv[0]))
                    self._convertCsvToh5(npCsv)
                except KeyError:
                    self._dfData = pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'data'))
            else:
                try:
                    self._dfData = pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'data'))
                    self._converth5ToCsv()
                except KeyError:
                    dfCsv = pd.read_hdf(self._rep + self._filename, 'csvFile')
                    npCsv = np.array(pd.DataFrame(dfCsv))
                    npCsv = npCsv.reshape((len(npCsv),))
                    self._dfData = pd.DataFrame(pd.read_csv(npCsv[0]))
            if self._verbose is True:
                print("The file already exists !!!")
        if self._verbose is True:
            print("Init Done !")

    def getDataXY(self):
        """
        Public Fonction
        :return: X and Y. X is a matrix containing profils NDVI generate. Y is label matrix
        """
        return self._generateXY()

    def getNoiseDataXY(self, noiseLevel, dictClassSystematicChange=None,seedNoise=0):
        """
        Public Fonction
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dictClassSystematicChange: Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :return: X and Y. X is a matrix containing profils NDVI generate. Y is label matrix with label noise
        """
        dfNoise = self._generateNoise(noiseLevel, dictClassSystematicChange,seedNoise)
        return self._generateXY(dfNoise)

    def getTestData(self):
        """
        Public Fonction
        :return: X and Y. X is a new matrix test containing profils NDVI generate. Y is label matrix
        """
        dfTest = self._generateTest()
        tmpdfData = self._dfData
        self._dfTest = dfTest
        self._dfData = dfTest
        (X, Y) = self._generateXY()
        self._dfData = tmpdfData
        return X, Y

    #  -----------------------------------------------------------------------------------------------------------------
    #  Intern (Private) Functions of this class.
    #  For normal use, don't use them.
    #  -----------------------------------------------------------------------------------------------------------------
    def _generateXY(self, dfLabel=None):
        """
        Intern Fonction
        :param dfLabel: DataFrame containing name of class for each pixel profil
        :return: X and Y. X is a new matrix test containing profils NDVI generate. Y is label matrix
        """
        if dfLabel is None:
            dfLabel = self._dfData
        Y = np.array(dfLabel.sort_values(by=['pixid'])['label']).reshape(6500, 1)
        X = np.array(self._dfData.sort_values(by=['pixid']).loc[:, 'd1':])
        X1 = np.ones((len(X), len(np.array(self._dfHeader.loc[0, :])[0])))
        for i in range(len(X)):
            X1[i] = X[i]
        return X1, Y

    def _generateNoise(self, noiseLevel, dictClassSystematicChange, seedNoise):
        """
        Intern Fonction
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dictClassSystematicChange: Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :return: DataFrame like this columns=['pixid', 'noisy', 'label']
        """
        name = self._genName(dictClassSystematicChange, noiseLevel)
        if not ReadGenerateData.getAlreadyGenNoise(self._filename, self._rep, name, self._csv):
            (noiseLevel, dfNoise, systematicChange) = GeneratorNoise.generatorNoisePerClass(self._filename, self._rep,
                                                                                            noiseLevel,
                                                                                            dictClassSystematicChange,
                                                                                            self._csv,
                                                                                            seed=seedNoise)
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep, noiseLevel, dfNoise, systematicChange,
                                                     self._csv, dictClassSystematicChange)
            if self._verbose is True:
                print("Generate Noise Done !")
        else:
            dfNoise = ReadGenerateData.getByNameNoise(self._filename, self._rep, name, self._csv)
            if self._verbose is True:
                print("Generate Noise Already Done !")
        return dfNoise

    def _generateTest(self):
        if self._csv is False:
            try:
                dfTest = pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'test'))
                if self._verbose is True:
                    print('Generate Test already Done !')
            except KeyError:
                (dfHeader, dfTest) = self._genData()
                WriteGenerateData.writeTest(self._filename, self._rep, dfTest, self._csv)
                if self._verbose is True:
                    print('Generate Test Done !')
        else:
            try:
                dfTest = pd.DataFrame(pd.read_csv(self._rep + 'test.csv'))
                if self._verbose is True:
                    print('Generate Test already Done !')
            except FileNotFoundError:
                (dfHeader, dfTest) = self._genData()
                WriteGenerateData.writeTest(self._filename, self._rep, dfTest, self._csv)
                if self._verbose is True:
                    print('Generate Test Done !')
        return dfTest

    def _genData(self):
        """
        Intern Fonction
        :return: 2 DataFrame Header and Data
        """
        (dfHeader, dfData) = GenerateData.generateData(seed=self._seedData)
        return dfHeader, dfData

    def _convertCsvToh5(self, npCsv):
        if self._verbose is True:
            print('Convert Csv To h5 start...')
        # Test if dataset Test exist
        try:
            dfTest = pd.DataFrame(pd.read_csv(self._rep + 'test.csv'))
        except FileNotFoundError:
            dfTest = None
        # Save old path
        tmpFileName = self._filename
        # Remove path of test.csv
        npCsv = npCsv[npCsv != self._rep + 'test.csv']
        self._filename = "test.h5"
        # Write dataset Data
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._rep, self._dfHeader, self._dfData, self._csv)
        # If dataset Test exist
        if dfTest is not None:
            # Write dataset Test and remove csv file
            WriteGenerateData.writeTest(self._filename, self._rep, dfTest, self._csv)
            os.remove(self._rep + 'test.csv')
        # Get content of LUT
        npLutCsv = np.array(pd.DataFrame(pd.read_csv(self._rep + 'lut.csv'))['dict'])
        # String to dict
        npLutCsv = [eval(i) for i in npLutCsv]
        # Remove path of data.csv
        npCsv = npCsv[npCsv != self._rep + 'data.csv']
        # Write noisy dataset
        for i in range(len(npCsv)):
            systematicChange = npCsv[i].split("/")[-1].split(".")[0]
            # Get noisy dataset
            dfNoise = pd.read_csv(npCsv[i])
            tmpstring = systematicChange.split('_')
            # Get noise level
            noiseLevel = int(tmpstring[1]) / 100
            if len(npLutCsv) == 0:
                WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep, noiseLevel, dfNoise,
                                                         systematicChange,
                                                         self._csv, None)
            else:
                WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep, noiseLevel, dfNoise,
                                                         systematicChange, self._csv, npLutCsv[i - 1])

            os.remove(self._rep + systematicChange + '.csv')
        os.remove(self._rep + 'data.csv')
        os.remove(self._rep + 'lut.csv')
        os.remove(self._rep + tmpFileName)
        os.rename(self._rep + self._filename, self._rep + tmpFileName)
        self._filename = tmpFileName
        if self._verbose is True:
            print('Convert Csv To h5 done !')

    def _converth5ToCsv(self):
        if self._verbose is True:
            print('Convert h5 To Csv start...')
        # Get name of dataset
        file = h5py.File(self._rep + self._filename, 'r')
        tmptab = list(file.keys())
        file.close()
        tmpDfNoise = []
        # Test if dataset Test exist
        try:
            dfTest = pd.read_hdf(self._rep + self._filename, tmptab[tmptab.index('test')])
            del tmptab[tmptab.index('test')]
        except ValueError:
            dfTest = None
        # Get dataset by name
        self._dfData = pd.read_hdf(self._rep + self._filename, tmptab[tmptab.index('data')])
        self._dfHeader = pd.read_hdf(self._rep + self._filename, tmptab[tmptab.index('header')])
        tmpDfHeaderNoise = np.array(
            pd.DataFrame(pd.read_hdf(self._rep + self._filename, tmptab[tmptab.index('headerNoise')])))
        # Reshape to horizontal array
        tmpDfHeaderNoise = tmpDfHeaderNoise.reshape((len(tmpDfHeaderNoise),))
        npLutCsv = np.array(pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'lut'))['dict'])
        # Remove already processed datasets
        del tmptab[tmptab.index('data')]
        del tmptab[tmptab.index('header')]
        del tmptab[tmptab.index('headerNoise')]
        del tmptab[tmptab.index('lut')]
        # Get noisy Dataframe
        for i in tmptab:
            tmpDfNoise.append(pd.read_hdf(self._rep + self._filename, i))
        tmpFileName = self._filename

        # Save old path
        self._filename = "test.h5"
        # Write dataset Data
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._rep, self._dfHeader, self._dfData, self._csv)
        # If dataset Test exist
        if dfTest is not None:
            # Write dataset Test and remove csv file
            WriteGenerateData.writeTest(self._filename, self._rep, dfTest, self._csv)
        # String to dict
        npLutCsv = [eval(i) for i in npLutCsv]
        # Write noisy dataset
        for i, j, k in zip(tmpDfHeaderNoise, tmpDfNoise, npLutCsv):
            systematicChange = i.split(".")[0]
            tmpstring = systematicChange.split('_')
            # Get noisy dataset
            dfNoise = j
            # Get noise level
            noiseLevel = int(tmpstring[1]) / 100
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep, noiseLevel, dfNoise, systematicChange,
                                                     self._csv, k)
        os.remove(self._rep + tmpFileName)
        os.rename(self._rep + self._filename, self._rep + tmpFileName)
        self._filename = tmpFileName
        if self._verbose is True:
            print('Convert h5 To Csv done !')

    @staticmethod
    def _genName(dictClass, noiseLevel):
        """
        Intern Fonction
        :param dictClass: dictClassSystematicChange: Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :param noiseLevel: Level of noise in %, format like that 0.05
        :return: Name of the dataset in file h5
        """
        name = ''
        if dictClass is not None:
            name = 'systematic_' + str(int(noiseLevel * 100)) + '_' + str(
                int(hashlib.md5(str(dictClass).encode("UTF-8")).hexdigest(), 16))
        else:
            name = 'random_' + str(int(noiseLevel * 100))
        return name