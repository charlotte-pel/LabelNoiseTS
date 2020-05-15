import matplotlib.pyplot as plt
import numpy as np


class Drawprofils:

    @staticmethod
    def drawProfilClass(className, dfHeader, dfData):
        """

        :param class_name:
        :param nb_class: number of class
        :param dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
        :param class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
        :param samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
               (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        :return: No return -> Draw graph
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
        plt.show()

    @staticmethod
    def drawProfilMeanClass(dfHeader, dfData):
        """

        :param class_name:
        :param nb_class: number of class
        :param dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
        :param class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
        :param samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
               (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        :return: No return -> Draw graph
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
        plt.show()

    @staticmethod
    def drawMeanProfilOneClass(className, dfHeader, dfData):
        """

        :param class_name:
        :param nb_class: number of class
        :param dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
        :param class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
        :param samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
               (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        :return: No return -> Draw graph
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
        plt.show()

    @staticmethod
    def drawMeanOneClass(className, dfHeader, dfData):
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
        plt.show()

    @staticmethod
    def draw20RandomProfilOneClass(className, dfHeader, dfData):
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
        plt.show()

    @staticmethod
    def draw20RandomIdProfilOneClass(className, dfHeader, dfData):
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
        plt.show()
