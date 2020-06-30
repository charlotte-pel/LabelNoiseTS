import matplotlib.pyplot as plt
import numpy as np


class Drawprofils:

    @staticmethod
    def drawProfilsOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw all profil for One class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        del dfTest['polid']
        del dfTest['pixid']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofil_' + className)
        else:
            plt.show()

    @staticmethod
    def drawProfilsMeanAllClass(dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw mean profil for all class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        nbClass = len(dfHeader)-2
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.groupby(['label']).mean()
        dfTest = dfTest.reset_index()
        dfTest = dfTest.set_index('label')
        del dfTest['polid']
        del dfTest['pixid']
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', kind='line')
        plt.title('NDVI Profils ' + str(nbClass) +' Classes')
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofilmean_10class')
        else:
            plt.show()

    @staticmethod
    def drawProfilMeanOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw mean profil for One class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.groupby(['label']).mean()
        dfTest = dfTest.reset_index()
        dfTest = dfTest.set_index('label')
        del dfTest['polid']
        del dfTest['pixid']
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofilmeanoneclass_' + className)
        else:
            plt.show()

    @staticmethod
    def drawMeanProfilsOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw Mean of each profil for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.groupby(['label', 'polid']).mean()
        dfTest = dfTest.reset_index()
        del dfTest['polid']
        del dfTest['pixid']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofilmeanid_' + className)
        else:
            plt.show()

    @staticmethod
    def draw20RandomProfilsOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw 20 random profils for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.reset_index()
        dfTest = dfTest.sample(n=20)
        del dfTest['polid']
        del dfTest['pixid']
        del dfTest['index']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofil20randomprofil_' + className)
        else:
            plt.show()

    @staticmethod
    def draw20RandomMeanProfilsOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw 20 random mean profils / mean of pixel of one polygon for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.groupby(['label', 'polid', 'pixid']).mean()
        dfTest = dfTest.reset_index()
        dfTest = dfTest.sample(n=20)
        del dfTest['polid']
        del dfTest['pixid']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('NDVI Profils ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofil20randomidMeanprofil_' + className)
        else:
            plt.show()

    @staticmethod
    def drawRandomOnePolyProfilOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw random pixel of one polygon for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.loc[(dfTest['polid'] == int(dfTest['polid'].sample(n=1)))]
        dfTest = dfTest.reset_index()
        del dfTest['polid']
        del dfTest['pixid']
        del dfTest['index']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotprofilrandomidprofil_' + className)
        else:
            plt.show()
