import matplotlib.pyplot as plt
import numpy as np


class Drawprofils:

    @staticmethod
    def drawProfilClass(className, dfHeader, dfData, vis=False,rep=''):
        """

        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param vis: True for save in file ot False for show plot
        :param rep: If vis == True -> name of the rep
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
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if vis is True:
            plt.savefig(rep+'plotprofil_'+className)
        else:
            plt.show()

    @staticmethod
    def drawProfilMeanClass(dfHeader, dfData,vis=False,rep=''):
        """

        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param vis: True for save in file ot False for show plot
        :param rep: If vis == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.groupby(['label']).mean()
        dfTest = dfTest.reset_index()
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        del dfTest['Build']
        del dfTest['Water']
        del dfTest['Wheat_Soy']
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', kind='line')
        plt.title('Profils NDVI simulés ' + 'all')
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis([0, 350, 0, 1])
        if vis is True:
            plt.savefig(rep+'plotprofilmean_allclass')
        else:
            plt.show()

    @staticmethod
    def drawMeanProfilOneClass(className, dfHeader, dfData,vis=False,rep=''):
        """

        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param vis: True for save in file ot False for show plot
        :param rep: If vis == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.groupby(['label', 'polid']).mean()
        dfTest = dfTest.reset_index()
        del dfTest['polid']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if vis is True:
            plt.savefig(rep+'plotprofilmeanid_'+className)
        else:
            plt.show()

    @staticmethod
    def drawMeanOneClass(className, dfHeader, dfData,vis=False,rep=''):
        """

        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param vis: True for save in file ot False for show plot
        :param rep: If vis == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.groupby(['label']).mean()
        dfTest = dfTest.reset_index()
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if vis is True:
            plt.savefig(rep+'plotprofilmeanoneclass_'+className)
        else:
            plt.show()

    @staticmethod
    def draw20RandomProfilOneClass(className, dfHeader, dfData,vis=False,rep=''):
        """

        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param vis: True for save in file ot False for show plot
        :param rep: If vis == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.reset_index()
        dfTest = dfTest.sample(n=20)  # , random_state=1)
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
        if vis is True:
            plt.savefig(rep+'plotprofil20randomprofil_'+className)
        else:
            plt.show()

    @staticmethod
    def draw20RandomIdProfilOneClass(className, dfHeader, dfData,vis=False,rep=''):
        """

        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param vis: True for save in file ot False for show plot
        :param rep: If vis == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        dfTest = dfData.loc[(dfData['label'] == className)]
        dfTest = dfTest.groupby(['label', 'polid']).mean()
        dfTest = dfTest.reset_index()
        dfTest = dfTest.sample(n=20)  # , random_state=1)
        del dfTest['polid']
        dfTest = dfTest.set_index('label')
        dfTest = dfTest.T
        dfTest.insert(0, 'dates', dates, True)
        dfTest.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Profils NDVI simulés ' + className)
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if vis is True:
            plt.savefig(rep+'plotprofil20randomidprofil_'+className)
        else:
            plt.show()
