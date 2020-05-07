import numpy as np

from SyntSITSLabelNoise.ReadGenerateData import *


class GeneratorNoise:
    """

    """

    @staticmethod
    def generatorNoisePerClass(filename, noiseLevel, class1=None, class2=None):
        (nbClass, dates, classNames, samplesClass, nbPixelClass,
         nbPolyClass) = ReadGenerateData.readGenerateDataH5DataFrame(filename)
        nbNoiseSamplesPerClass = np.floor(np.multiply(noiseLevel, nbPixelClass))
        dfNoise = samplesClass[['pixid', 'label']]
        dfNoise.insert(1, "noisy", False, True)
        tmpDfNoise = []
        if (class1 is not None) & (class2 is not None):
            classNames = [class1, class2]
        for i in classNames:
            tmpIdTab = []
            for j in range(0, int(nbNoiseSamplesPerClass[classNames.index(i)]), 10):
                nbIdRand = np.random.randint(1, nbPolyClass[classNames.index(i)])
                while nbIdRand in tmpIdTab:
                    nbIdRand = np.random.randint(1, nbPolyClass[classNames.index(i)])
                tmpPixTab = samplesClass.loc[(samplesClass["label"] == i) & (samplesClass["polid"] == nbIdRand)]
                tmpPixTab = np.array(tmpPixTab['pixid'])
                newClass = np.random.choice(classNames, 1)
                while newClass is i:
                    newClass = np.random.choice(classNames)
                newClass = newClass[0]
                if j < nbNoiseSamplesPerClass[0] - 5:
                    for k in tmpPixTab:
                        tmpDfNoise.append((k, True, newClass))
                else:
                    for k in range(0, int(nbNoiseSamplesPerClass[0] % (j))):
                        tmpDfNoise.append((tmpPixTab[k], True, newClass))
                tmpIdTab.append(nbIdRand)
        #print(np.array(tmpDfNoise).shape)
        #print(nbNoiseSamplesPerClass[0] * nbClass)
        tmpDfNoise = pd.DataFrame(tmpDfNoise, columns=['pixid', 'noisy', 'label'])
        dfNoise = dfNoise[['pixid']].merge(tmpDfNoise, 'right').combine_first(dfNoise).astype(dfNoise.dtypes)
        return dfNoise











