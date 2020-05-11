from SyntSITSLabelNoise.GenerateData import *
from SyntSITSLabelNoise.GeneratorNoise import *
from SyntSITSLabelNoise.InitParamValues import *
from SyntSITSLabelNoise.WriteGenerateData import *
import numpy as np


class GeneratorData:

    def __init__(self, filename):
        """

        :param filename: Name of the file h5
        """
        print("Init Start !")
        (dfHeader, dfData) = self._genData()
        self._dfHeader = dfHeader
        self._dfData = dfData
        self._filename = filename
        self._dfTest = None
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._dfHeader, self._dfData)
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
        if not ReadGenerateData.getAlreadyGenNoise(self._filename, name):
            (noiseLevel, dfNoise, systematicChange) = GeneratorNoise.generatorNoisePerClass(self._filename, noiseLevel,
                                                                                            dictClassSystematicChange)
            WriteGenerateData.writeGenerateNoisyData(self._filename, noiseLevel, dfNoise, systematicChange)
            print("Generate Noise Done !")
        else:
            dfNoise = ReadGenerateData.getByNameNoise(self._filename, name)
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
        X = np.array(self._dfData.sort_values(by=['pixid'])['profil'])
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
        dates = InitParamValues.generateDates()
        (dfHeader, dfData) = GenerateData.generateData(class_names, param_val, dates)
        return dfHeader, dfData

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