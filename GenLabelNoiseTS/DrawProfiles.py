import matplotlib.pyplot as plt
import numpy as np


class DrawProfiles:

    @staticmethod
    def drawProfilesOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw all profile for One class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        tmpDfData = dfData.loc[(dfData['label'] == className)]
        del tmpDfData['polid']
        del tmpDfData['pixid']
        tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotProfilesOneClass_' + className)
        else:
            plt.show()

    @staticmethod
    def drawProfilesMeanAllClass(dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw mean profile for all class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        nbClass = len(dfHeader)-2
        dates = np.array(dfHeader.loc[0, :])[0]
        tmpDfData = dfData
        del tmpDfData['polid']
        del tmpDfData['pixid']
        tmpDfData = dfData.groupby(['label']).mean()
        # tmpDfData = tmpDfData.reset_index()
        # tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', kind='line')
        plt.title('Simulated NDVI profiles ' + str(nbClass) +' Classes')
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotProfilesMeanAllClass_'+str(nbClass)+'class')
        else:
            plt.show()

    @staticmethod
    def drawProfileMeanOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw mean profile for One class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        tmpDfData = dfData.loc[(dfData['label'] == className)]
        del tmpDfData['polid']
        del tmpDfData['pixid']
        tmpDfData = tmpDfData.groupby(['label']).mean()
        tmpDfData = tmpDfData.reset_index()
        tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotProfileMeanOneClass_' + className)
        else:
            plt.show()

    @staticmethod
    def drawMeanProfilesOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw Mean of each profile for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        tmpDfData = dfData.loc[(dfData['label'] == className)]
        del tmpDfData['pixid']
        tmpDfData = tmpDfData.groupby(['label', 'polid']).mean()
        tmpDfData = tmpDfData.reset_index()
        del tmpDfData['polid']
        tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotMeanProfilesOneClass_' + className)
        else:
            plt.show()

    @staticmethod
    def draw20RandomProfilesOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw 20 random profiles for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        tmpDfData = dfData.loc[(dfData['label'] == className)]
        tmpDfData = tmpDfData.reset_index()
        tmpDfData = tmpDfData.sample(n=20)
        del tmpDfData['polid']
        del tmpDfData['pixid']
        del tmpDfData['index']
        tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plot20RandomProfilesOneClass_' + className)
        else:
            plt.show()

    @staticmethod
    def draw20RandomMeanProfilesOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
        """
        Draw 20 random mean profiles / mean of pixel of one polygon for one class
        :param className: Name of the class
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param saveFile: True for save in file ot False for show plot
        :param rep: If saveFile == True -> name of the rep
        :return: No return -> Draw graph or save in file
        """
        dates = np.array(dfHeader.loc[0, :])[0]
        tmpDfData = dfData.loc[(dfData['label'] == className)]
        tmpDfData = tmpDfData.groupby(['label', 'polid', 'pixid']).mean()
        tmpDfData = tmpDfData.reset_index()
        tmpDfData = tmpDfData.sample(n=20)
        del tmpDfData['polid']
        del tmpDfData['pixid']
        tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plot20RandomMeanProfilesOneClass_' + className)
        else:
            plt.show()

    @staticmethod
    def drawRandomOnePolyProfileOneClass(className, dfHeader, dfData, saveFile=False, rep=''):
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
        tmpDfData = dfData.loc[(dfData['label'] == className)]
        print(tmpDfData)
        #TODO Fix bug first launch
        tmpDfData = tmpDfData.loc[(tmpDfData['polid'] == int(tmpDfData['polid'].sample(n=1)))]
        tmpDfData = tmpDfData.reset_index()
        del tmpDfData['polid']
        del tmpDfData['pixid']
        del tmpDfData['index']
        tmpDfData = tmpDfData.set_index('label')
        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        if saveFile is True:
            plt.savefig(rep + 'plotRandomOnePolyProfileOneClass_' + className)
        else:
            plt.show()
