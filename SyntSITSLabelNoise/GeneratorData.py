import h5py

from SyntSITSLabelNoise.GenerateData import *
from SyntSITSLabelNoise.GeneratorNoise import *
from SyntSITSLabelNoise.InitParamValues import *
from SyntSITSLabelNoise.WriteGenerateData import *
import numpy as np


class GeneratorData:

    def __init__(self, filename,csv,rep=''):
        """

        :param filename: Name of the file h5
        """
        print("Init Start !")
        self._rep = rep
        self._filename = filename
        self._dfTest = None
        self._csv = csv
        if not os.path.isfile(self._rep+self._filename):
            (dfHeader, dfData) = self._genData()
            self._dfHeader = dfHeader
            self._dfData = dfData
            WriteGenerateData.writeGenerateDataToH5(self._filename, self._rep,self._dfHeader, self._dfData,self._csv)
        else:
            self._dfHeader = pd.read_hdf(self._rep+self._filename,'header')
            if csv is False:
                try:
                    dfCsv = pd.read_hdf(self._rep+self._filename, 'csvFile')
                    npCsv = np.array(pd.DataFrame(dfCsv)).reshape((2,))
                    self._dfData = pd.DataFrame(pd.read_csv(npCsv[0]))
                    self._dfHeader = pd.DataFrame(pd.read_hdf(self._rep+self._filename,'header'))
                    self.convertCsvToh5(npCsv)
                except KeyError:
                    self._dfData = pd.DataFrame(pd.read_hdf(self._rep+self._filename, 'data'))
            else:
                self._dfData = pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'data'))
                self._dfHeader = pd.DataFrame(pd.read_hdf(self._rep + self._filename, 'header'))
                self.converth5ToCsv()
            print("The file already exists !!!")
        print("Init Done !")

    def getDataXY(self):
        """
        Public Fonction
        :return: X and Y. X is a matrix containing profils NDVI generate. Y is label matrix
        """
        print("Generate Data Done !")
        return self._generateXY()

    def getNoiseDataXY(self, noiseLevel, dictClassSystematicChange=None):
        """
        Public Fonction
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dictClassSystematicChange: Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :return: X and Y. X is a matrix containing profils NDVI generate. Y is label matrix with label noise
        """
        dfNoise = self._generateNoise(noiseLevel, dictClassSystematicChange)
        return self._generateXY(dfNoise)

    def getTestData(self):
        """
        Public Fonction
        :return: X and Y. X is a new matrix test containing profils NDVI generate. Y is label matrix
        """
        (dfHeader, dfTest) = self._genData()
        WriteGenerateData.writeTest(self._filename, dfTest)
        tmpdfData = self._dfData
        self._dfTest = dfTest
        self._dfData = dfTest
        (X, Y) = self._generateXY()
        self._dfData = tmpdfData
        print("Generate Test-Data Done !")
        return X, Y

    #  -----------------------------------------------------------------------------------------------------------------
    #  Intern (Private) Functions of this class.
    #  For normal use, don't use them.
    #  -----------------------------------------------------------------------------------------------------------------

    def _generateNoise(self, noiseLevel, dictClassSystematicChange=None):
        """
        Intern Fonction
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dictClassSystematicChange: Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :return: DataFrame like this columns=['pixid', 'noisy', 'label']
        """
        name = self._genName(dictClassSystematicChange, noiseLevel)
        if not ReadGenerateData.getAlreadyGenNoise(self._filename, self._rep,name,self._csv):
            (noiseLevel, dfNoise, systematicChange) = GeneratorNoise.generatorNoisePerClass(self._filename, self._rep,noiseLevel,
                                                                                            dictClassSystematicChange,self._csv)
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep,noiseLevel, dfNoise, systematicChange,self._csv)
            print("Generate Noise Done !")
        else:
            dfNoise = ReadGenerateData.getByNameNoise(self._filename,self._rep, name,self._csv)
            print("Generate Noise Already Done !")
        return dfNoise

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

    def _genData(self):
        """
        Intern Fonction
        :return: 2 DataFrame Header and Data
        """
        (param_val, class_names) = InitParamValues.initParamValues(3)
        class_names = np.array(class_names)
        dates = InitParamValues.generateDates()
        (dfHeader, dfData) = GenerateData.generateData(class_names, param_val, dates)
        return dfHeader, dfData

    def convertCsvToh5(self,npCsv):
        print('Convert Csv To h5 start...')
        self._filename = "test.h5"
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._rep, self._dfHeader, self._dfData, self._csv)
        for i in range(1,len(npCsv)):
            dfNoise = pd.read_csv(npCsv[i])
            tmpstring = npCsv[i].split("/")[1].split(".")[0].split('_')
            noiseLevel = int(tmpstring[-1])/100
            del tmpstring[-1]
            del tmpstring[0]
            systematicChange = ''
            for j in tmpstring:
                systematicChange = systematicChange+j+'_'
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep, noiseLevel, dfNoise, systematicChange,
                                                     self._csv)
        print('Convert Csv To h5 done !')

    def converth5ToCsv(self):
        file = h5py.File(self._rep+self._filename, 'r')
        tmptab = list(file.keys())
        file.close()
        tmpDfNoise = []
        self._dfData = pd.read_hdf(self._rep+self._filename,tmptab[0])
        self._dfHeader = pd.read_hdf(self._rep + self._filename, tmptab[1])
        for i in tmptab[3:]:
            tmpDfNoise.append(pd.read_hdf(self._rep + self._filename, i))
        tmpDfHeaderNoise = np.array(pd.DataFrame(pd.read_hdf(self._rep + self._filename, tmptab[2])))
        tmpDfHeaderNoise = tmpDfHeaderNoise.reshape((len(tmpDfHeaderNoise),))
        self._filename = "test.h5"
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._rep, self._dfHeader, self._dfData, self._csv)
        for i,j in zip(tmpDfHeaderNoise,tmpDfNoise):
            tmpstring = i.split(".")[0].split('_')
            print(tmpstring)
            dfNoise = j
            noiseLevel = int(tmpstring[-1])/100
            del tmpstring[-1]
            del tmpstring[0]
            systematicChange = ''
            for j in tmpstring:
                systematicChange = systematicChange + j + '_'
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._rep, noiseLevel, dfNoise, systematicChange,
                                                     self._csv)



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
            for i in dictClass.items():
                classNames = i
                name = name + classNames[0] + 'To' + str(classNames[1]) + '_'
            name = 'noiseData_' + name
        else:
            name = 'noiseData_'
        return name + str(int(noiseLevel * 100))