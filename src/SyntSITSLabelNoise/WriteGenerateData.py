from src.SyntSITSLabelNoise.GenerateData import *

class WriteGenerateData:

    @staticmethod
    def writeGenerateDataToH5(filename,dfHeader, dfData):
        hdf = pd.HDFStore(filename)
        hdf.put('header', dfHeader)
        hdf.put('data', dfData)
        hdf.close()