import pandas as pd


class ReadGenerateData:

    @staticmethod
    def readGenerateDataH5DataFrame(filename):
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
        # Doesn't work in all cases: if size of poly aren't the same. The size is fixed -> samplesClass['polid'] == 1
        nbPolyClass = [len(samplesClass.loc[(samplesClass['label'] == i)]) / len(
            samplesClass.loc[(samplesClass['label'] == i) & (samplesClass['polid'] == 1)]) for i in classNames]
        return nbClass, dates, classNames, samplesClass, nbPixelClass, nbPolyClass
