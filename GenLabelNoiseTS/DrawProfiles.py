import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


class DrawProfiles:

    @staticmethod
    def drawProfiles(dfHeader, dfData, typePlot, className=None, nbProfile=20, dir=None):
        """
        Drawn data plot.
        :param dfHeader: DataFrame contain Header
        :param dfData: DataFrame contain Data
        :param typePlot: typePlot = 'all' or 'mean' or 'random' or 'randomPoly'
        :param className: None or Name of the class
        :param nbProfile: Number of profile for typePlot = random
        :param dir: If dir = None show plot, if dir is specify plot will be save in dir.
        :return: No return -> Draw graph or save in file
        """
        if typePlot not in ['all', 'mean', 'random', 'randomPoly']:
            print('typePlot ERROR')
            sys.exit(0)

        if dir is not None:
            dir = Path(dir)

        dates = np.array(dfHeader.loc[0, :])[0]
        nbClass = len(dfHeader) - 2

        tmpDfData = dfData
        if typePlot == 'randomPoly':
            del tmpDfData['pixid']
        else:
            del tmpDfData['polid']
            del tmpDfData['pixid']

        if typePlot == 'mean' and className is None:
            tmpDfData = dfData.groupby(['label']).mean()
        elif typePlot == 'mean' and className is not None:
            tmpDfData = dfData.loc[(dfData['label'] == className)].groupby(['label']).mean()
        elif typePlot == 'all' and className is not None:
            tmpDfData = dfData.loc[(dfData['label'] == className)]
            tmpDfData = tmpDfData.set_index('label')
        elif typePlot == 'random' and className is not None:
            tmpDfData = dfData.loc[(dfData['label'] == className)]
            tmpDfData = tmpDfData.set_index('label')
            tmpDfData = tmpDfData.sample(n=nbProfile)
        elif typePlot == 'randomPoly' and className is not None:
            tmpDfData = dfData.loc[(dfData['label'] == className)]
            tmpDfData = tmpDfData.loc[(tmpDfData['polid'] == int(tmpDfData['polid'].sample(n=1)))]
            tmpDfData = tmpDfData.set_index('label')
            del tmpDfData['polid']
        else:
            print('Error className was not specify for typePlot = all/random/randomPoly')

        tmpDfData = tmpDfData.T
        tmpDfData.insert(0, 'dates', dates, True)
        tmpDfData.plot(x='dates', y=className, kind='line', legend=False)
        if typePlot == 'mean' and className is None:
            plt.title('Simulated NDVI profiles ' + str(nbClass) + ' Classes')
        else:
            plt.title('Simulated NDVI profiles ' + className)
        plt.grid()
        plt.xlabel('DoY')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])

        if dir is None:
            plt.show()
        else:
            if typePlot == 'mean' and className is None:
                plt.savefig(dir / ('plotProfilesMeanAllClass_' + str(nbClass) + 'class'))
            elif typePlot == 'mean' and className is not None:
                plt.savefig(dir / ('plotProfileMeanOneClass_' + className))
            elif typePlot == 'all' and className is not None:
                plt.savefig(dir / ('plotProfilesOneClass_' + className))
            elif typePlot == 'random' and className is not None:
                plt.savefig(dir / ('plot20RandomProfilesOneClass_' + className))
            elif typePlot == 'randomPoly' and className is not None:
                plt.savefig(dir / ('plotRandomOnePolyProfileOneClass_' + className))
