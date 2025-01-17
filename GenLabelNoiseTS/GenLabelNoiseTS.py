import os

import h5py

from GenLabelNoiseTS.DrawProfiles import *
from GenLabelNoiseTS.GeneratorLabelNoise import *
from GenLabelNoiseTS.GeneratorNDVIProfiles import *
from GenLabelNoiseTS.WriteGenerateData import *


class GenLabelNoiseTS:

    def __init__(self, filename, dir='', classList=None, pathInitFile=None, seedData=None, csv=False,
                 verbose=False):
        """

        :param filename: Name of the file h5
        :param dir: name of the dir
        :param classList: List of class names for generate data.
        :param pathInitFile: Path to initParamFile
        :param seedData: seed for randomState
        :param csv: Csv Option True or False
        :param verbose: Show print : True or False
        """

        self._classList = classList

        if seedData is None:
            seedData = np.random.randint(100000)

        self._verbose = verbose
        if self._verbose is True:
            print("Init Start !")
        self._dir = Path(dir)
        self._filename = filename
        self._dfTest = None
        self._csv = csv
        self._seedData = seedData
        # Test if not file exist
        if not os.path.isfile(self._dir / self._filename):
            # Path to initFilename
            if pathInitFile is None:
                self._initFilename = Path('init_param_file.csv')
            else:
                self._initFilename = Path(pathInitFile)
            (dfHeader, dfData) = self._genData()
            self._dfHeader = dfHeader
            self._noPixPerPolid = self._getDfNoPixPerPolidList()
            self._dfData = dfData
            WriteGenerateData.writeGenerateDataToH5(self._filename, self._dir, self._dfHeader, self._dfData, self._csv)
            if self._verbose is True:
                print("Generate Data Done !")

        else:
            # File exist
            self._dfHeader = pd.DataFrame(pd.read_hdf(self._dir / self._filename, 'header'))
            self._noPixPerPolid = self._getDfNoPixPerPolidList()
            # Test if convert is needed
            if csv is False:
                try:
                    dfCsv = pd.read_hdf(self._dir / self._filename, 'csvFile')
                    npCsv = np.array(pd.DataFrame(dfCsv))
                    npCsv = npCsv.reshape((len(npCsv),))
                    self._dfData = pd.DataFrame(pd.read_csv(self._dir / npCsv[0]))
                    self._convertCsvToh5(self._dir / npCsv)
                except KeyError:
                    self._dfData = pd.DataFrame(pd.read_hdf(self._dir / self._filename, 'data'))
            else:
                try:
                    self._dfData = pd.DataFrame(pd.read_hdf(self._dir / self._filename, 'data'))
                    self._converth5ToCsv()
                except KeyError:
                    dfCsv = pd.read_hdf(self._dir / self._filename, 'csvFile')
                    npCsv = np.array(pd.DataFrame(dfCsv))
                    npCsv = npCsv.reshape((len(npCsv),))
                    self._dfData = pd.DataFrame(pd.read_csv(self._dir / npCsv[0]))
            if self._verbose is True:
                print("The file already exists !!!")
        (_, Y) = self._generateXY()
        self._matrixClassInt = pd.DataFrame([np.arange(0, len(np.unique(Y)))], columns=np.unique(Y))
        if self._verbose is True:
            print("Init Done !")

    #  -----------------------------------------------------------------------------------------------------------------
    #  External (Public) Functions of this class.
    #  For normal use.
    #  -----------------------------------------------------------------------------------------------------------------

    def getDataXY(self):
        """
        Public function
        :return: X and Y. X is a matrix containing profils NDVI generate. Y is label matrix
        """
        (X, Y) = self._generateXY()
        return self._strClassNamesToInt(X, Y)

    def getNoiseDataXY(self, noiseLevel, dictClassSystematicChange=None, seedNoise=None):
        """
        Public function
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dictClassSystematicChange: None = random noise if dict is specify == Systematic change
                Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :param seedNoise: seed for randomState
        :return: X and Y. X is a matrix containing profils NDVI generate. Y is label matrix with label noise
        """
        if seedNoise is None:
            seedNoise = np.random.randint(100000)
        dfNoise = self._generateNoise(noiseLevel, dictClassSystematicChange, seedNoise)
        (X, Y) = self._generateXY(dfNoise)
        return self._strClassNamesToInt(X, Y)

    def getTestData(self, otherPath=None):
        """
        Public function
        :return: X and Y. X is a new matrix test containing profils NDVI generate. Y is label matrix
        """
        if otherPath is not None:
            otherPath = Path(otherPath)
        dfTest = self._generateTest(otherPath)
        tmpdfData = self._dfData
        self._dfTest = dfTest
        self._dfData = dfTest
        (X, Y) = self._generateXY()
        self._dfData = tmpdfData
        return self._strClassNamesToInt(X, Y)

    def getDfHeader(self):
        """
        Getter for dfHeader
        :return: The header dataframe
        """
        return self._dfHeader

    def getDfData(self):
        """
        Getter for dfData
        :return: The data dataframe
        """
        return self._dfData

    def getSeed(self):
        return self.getSeedList()['dataSeed']

    def getSeedList(self):
        return eval(str(self._dfHeader[0][1]))

    def getMatrixClassInt(self):
        return self._matrixClassInt

    @staticmethod
    def getNoiseMatrix(Y_True, Y_Noise):
        """
        Public function
        :param Y_True: Array with originals labels
        :param Y_Noise: Array with noisy labels
        :return: Noise Matrix with shape(noUniqueClass_Y_True,noUniqueClass_Y_Noise)
        """
        noiseMatrix = np.zeros((len(np.unique(Y_True)), len(np.unique(Y_True))), dtype=int)
        for i, j in zip(Y_True, Y_Noise):
            noiseMatrix[i, j] += 1
        return noiseMatrix

    #  -----------------------------------------------------------------------------------------------------------------
    #  Visualisation Functions of this class.
    #  -----------------------------------------------------------------------------------------------------------------
    def visualisation(self, typePlot, className=None, noProfile=20, dir=None):
        """
        Visualisation function to draw plot
        :param typePlot: typePlot = 'all' or 'mean' or 'random' or 'randomPoly'
        :param className: None or Name of the class
        :param noProfile: Number of profile for typePlot = random
        :param dir: If dir = None show plot, if dir is specify plot will be save in dir
        :return: No return -> Draw graph or save in file depending of dir value
        """
        if dir is not None:
            dir = Path(dir)
        DrawProfiles.drawProfiles(self._dfHeader.copy(), self._dfData.copy(), typePlot=typePlot, className=className,
                                  noProfile=noProfile, dir=dir)

    def defaultVisualisation(self, dir=None):
        """
        Draw by default one plot by class to show all profiles and one mean plot
        :param dir: If dir = None show plot, if dir is specify plot will be save in dir
        :return: No return -> Draw graph or save in file depending of dir value
        """
        if dir is not None:
            dir = Path(dir)
        DrawProfiles.drawProfiles(self._dfHeader.copy(), self._dfData.copy(), typePlot='mean', className=None, dir=dir)
        for i in self._classList:
            DrawProfiles.drawProfiles(self._dfHeader.copy(), self._dfData.copy(), typePlot='all', className=i, dir=dir)

    #  -----------------------------------------------------------------------------------------------------------------
    #  Intern (Private) Functions of this class.
    #  For normal use, don't use them.
    #  -----------------------------------------------------------------------------------------------------------------
    def _generateXY(self, dfLabel=None):
        """
        Intern function
        :param dfLabel: DataFrame containing name of class for each pixel profil
        :return: X and Y. X is a new matrix containing profils NDVI generate. Y is label matrix
        """
        if dfLabel is None:
            dfLabel = self._dfData
        Y = np.array(dfLabel.sort_values(by=['pixid'])['label']).reshape(
            len(np.array(dfLabel.sort_values(by=['pixid'])['label'])), 1)
        X = np.array(self._dfData.sort_values(by=['pixid']).loc[:, 'd1':])
        X1 = np.ones((len(X), len(np.array(self._dfHeader.loc[0, :])[0])))
        for i in range(len(X)):
            X1[i] = X[i]
        return X1, Y

    def _generateNoise(self, noiseLevel, dictClassSystematicChange, seedNoise):
        """
        Intern function
        :param noiseLevel: Level of noise in %, format like that 0.05
        :param dictClassSystematicChange: Dictionary format like that {'Wheat': 'Barley', 'Barley': 'Soy'}
        :param seedNoise: seed for randomState
        :return: DataFrame like this columns=['pixid', 'noisy', 'label']
        """
        name = self._genName(dictClassSystematicChange, noiseLevel)
        if not ReadGenerateData.getAlreadyGenNoise(self._filename, self._dir, name, self._csv):
            generatorNoise = GeneratorLabelNoise(filename=self._filename, dir=self._dir,
                                                 noiseLevel=noiseLevel, seed=seedNoise,
                                                 dfNoPixPerPolidList=self._noPixPerPolid,
                                                 dictClass=dictClassSystematicChange,
                                                 csv=self._csv)
            (noiseLevel, dfNoise, systematicChange) = generatorNoise.generatorNoisePerClass()
            tmpDictSeed = self.getSeedList()

            # Add new seedNoise to Header.
            if dictClassSystematicChange is None:
                tmpDictSeed[str(noiseLevel) + '_Random'] = seedNoise
            else:
                tmpDictSeed[str(noiseLevel) + '_' + str(dictClassSystematicChange)] = seedNoise

            self._dfHeader.loc[1:1, 0:0] = np.array(tmpDictSeed)
            WriteGenerateData.updateDfHeader(self._filename, self._dir, self._dfHeader)
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._dir, noiseLevel, dfNoise, systematicChange,
                                                     self._csv, dictClassSystematicChange)
            del generatorNoise
            if self._verbose is True:
                print("Generate Noise Done !")
        else:
            dfNoise = ReadGenerateData.getByNameNoise(self._filename, self._dir, name, self._csv)
            if self._verbose is True:
                print("Generate Noise Already Done !")
        return dfNoise

    def _generateTest(self, otherPath):
        """
        Intern function
        :return: DataFrame contain test dataset
        """
        if self._csv is False:
            try:
                dfTest = pd.DataFrame(pd.read_hdf(self._dir / self._filename, 'test'))
                if self._verbose is True:
                    print('Generate Test already Done !')
            except KeyError:
                (dfHeader, dfTest) = self._genData()
                WriteGenerateData.writeTest(self._filename, self._dir, dfTest, self._csv)
                if self._verbose is True:
                    print('Generate Test Done !')
        else:
            if otherPath is None:
                dirPath = self._dir
            else:
                dirPath = Path(otherPath)
            try:
                dfTest = pd.DataFrame(pd.read_csv(dirPath / 'test.csv'))
                if self._verbose is True:
                    print('Generate Test already Done !')
            except FileNotFoundError:
                (dfHeader, dfTest) = self._genData()
                WriteGenerateData.writeTest(self._filename, dirPath, dfTest, self._csv)
                if self._verbose is True:
                    print('Generate Test Done !')
        return dfTest

    def _genData(self):
        """
        Intern function
        :return: 2 DataFrame Header and Data
        """
        (dfHeader, dfData) = GeneratorNDVIProfiles.generatorNDVIProfiles(seed=self._seedData,
                                                                         initFilename=self._initFilename,
                                                                         classList=self._classList)
        return dfHeader, dfData

    def _strClassNamesToInt(self, X, Y, seed=None):
        Yint = np.array([int(self._matrixClassInt[i[0]]) for i in Y]).reshape(
            len(np.array([int(self._matrixClassInt[i[0]]) for i in Y])), 1)

        # Shuffle the X et Y Matrix
        if seed is None:
            seed = np.random.randint(100000)
        randomStateGen = np.random.RandomState(seed)
        X = randomStateGen.permutation(X)
        randomStateGen = np.random.RandomState(seed)
        Yint = randomStateGen.permutation(Yint)

        return X, Yint

    def _convertCsvToh5(self, npCsv):
        """
        Intern function for convert Csv file save system to H5 file save system
        :param npCsv: Numpy array contain path of csv file
        :return: None
        """
        if self._verbose is True:
            print('Convert Csv To h5 start...')
        # Test if dataset Test exist
        try:
            dfTest = pd.DataFrame(pd.read_csv(self._dir / 'test.csv'))
        except FileNotFoundError:
            dfTest = None
        # Save old path
        tmpFileName = self._filename
        # Remove path of test.csv
        npCsv = npCsv[npCsv != self._dir / 'test.csv']
        self._filename = "test.h5"
        # Write dataset Data
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._dir, self._dfHeader, self._dfData, self._csv)
        # If dataset Test exist
        if dfTest is not None:
            # Write dataset Test and remove csv file
            WriteGenerateData.writeTest(self._filename, self._dir, dfTest, self._csv)
            os.remove(self._dir / 'test.csv')
        # Get content of LUT
        npLutCsv = np.array(pd.DataFrame(pd.read_csv(self._dir / 'lut.csv'))['dict'])
        # String to dict
        npLutCsv = [eval(i) for i in npLutCsv]
        # Remove path of data.csv
        npCsv = npCsv[npCsv != self._dir / 'data.csv']
        # Write noisy dataset
        for i in range(len(npCsv)):
            systematicChange = npCsv[i].split("/")[-1].split(".")[0]
            # Get noisy dataset
            dfNoise = pd.read_csv(npCsv[i])
            tmpstring = systematicChange.split('_')
            # Get noise level
            noiseLevel = int(tmpstring[1]) / 100
            if len(npLutCsv) == 0:
                WriteGenerateData.writeGenerateNoisyData(self._filename, self._dir, noiseLevel, dfNoise,
                                                         systematicChange,
                                                         self._csv, None)
            else:
                WriteGenerateData.writeGenerateNoisyData(self._filename, self._dir, noiseLevel, dfNoise,
                                                         systematicChange, self._csv, npLutCsv[i - 1])

            os.remove(self._dir / (systematicChange + '.csv'))
        os.remove(self._dir / 'data.csv')
        os.remove(self._dir / 'lut.csv')
        os.remove(self._dir / tmpFileName)
        os.rename(self._dir / self._filename, self._dir / tmpFileName)
        self._filename = tmpFileName
        if self._verbose is True:
            print('Convert Csv To h5 done !')

    def _converth5ToCsv(self):
        """
        Intern function for convert H5 file save system to CSV file save system
        :return: None
        """
        if self._verbose is True:
            print('Convert h5 To Csv start...')
        # Get name of dataset
        file = h5py.File(self._dir / self._filename, 'r')
        tmptab = list(file.keys())
        file.close()
        tmpDfNoise = []
        # Test if dataset Test exist
        try:
            dfTest = pd.read_hdf(self._dir / self._filename, tmptab[tmptab.index('test')])
            del tmptab[tmptab.index('test')]
        except ValueError:
            dfTest = None
        # Get dataset by name
        self._dfData = pd.read_hdf(self._dir / self._filename, tmptab[tmptab.index('data')])
        self._dfHeader = pd.read_hdf(self._dir / self._filename, tmptab[tmptab.index('header')])
        self._noPixPerPolid = self._getDfNoPixPerPolidList()
        tmpDfHeaderNoise = np.array(
            pd.DataFrame(pd.read_hdf(self._dir / self._filename, tmptab[tmptab.index('headerNoise')])))
        # Reshape to horizontal array
        tmpDfHeaderNoise = tmpDfHeaderNoise.reshape((len(tmpDfHeaderNoise),))
        npLutCsv = np.array(pd.DataFrame(pd.read_hdf(self._dir / self._filename, 'lut'))['dict'])
        # Remove already processed datasets
        del tmptab[tmptab.index('data')]
        del tmptab[tmptab.index('header')]
        del tmptab[tmptab.index('headerNoise')]
        del tmptab[tmptab.index('lut')]
        # Get noisy Dataframe
        for i in tmptab:
            tmpDfNoise.append(pd.read_hdf(self._dir / self._filename, i))
        tmpFileName = self._filename

        # Save old path
        self._filename = "test.h5"
        # Write dataset Data
        WriteGenerateData.writeGenerateDataToH5(self._filename, self._dir, self._dfHeader, self._dfData, self._csv)
        # If dataset Test exist
        if dfTest is not None:
            # Write dataset Test and remove csv file
            WriteGenerateData.writeTest(self._filename, self._dir, dfTest, self._csv)
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
            WriteGenerateData.writeGenerateNoisyData(self._filename, self._dir, noiseLevel, dfNoise, systematicChange,
                                                     self._csv, k)
        os.remove(self._dir / tmpFileName)
        os.rename(self._dir / self._filename, self._dir / tmpFileName)
        self._filename = tmpFileName
        if self._verbose is True:
            print('Convert h5 To Csv done !')

    @staticmethod
    def _genName(dictClass, noiseLevel):
        """
        Intern function
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

    def _getDfNoPixPerPolidList(self):
        return pd.DataFrame([[i[0], i[1][1]] for i in np.array(self._dfHeader.loc[2:, 0])],
                            columns=['label', 'noPixelPerPolid'])
