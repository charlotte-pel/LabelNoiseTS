import matplotlib.pyplot as plt
import numpy as np
import sys


class DrawProfiles:

    @staticmethod
    def drawProfiles(dfHeader, dfData, typePlot, className=None, nbProfile=20, rep=None):
        if typePlot not in ['all', 'mean', 'random', 'randomPoly']:
            print('typePlot ERROR')
            sys.exit(0)

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

        if rep is None:
            plt.show()
        else:
            if typePlot == 'mean' and className is None:
                plt.savefig(rep + 'plotProfilesMeanAllClass_' + str(nbClass) + 'class')
            elif typePlot == 'mean' and className is not None:
                plt.savefig(rep + 'plotProfileMeanOneClass_' + className)
            elif typePlot == 'all' and className is not None:
                plt.savefig(rep + 'plotProfilesOneClass_' + className)
            elif typePlot == 'random' and className is not None:
                plt.savefig(rep + 'plot20RandomProfilesOneClass_' + className)
            elif typePlot == 'randomPoly' and className is not None:
                plt.savefig(rep + 'plotRandomOnePolyProfileOneClass_' + className)
