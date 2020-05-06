import numpy as np

from src.SyntSITSLabelNoise.ReadGenerateData import *


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
        print(np.array(tmpDfNoise).shape)
        print(nbNoiseSamplesPerClass[0] * nbClass)
        tmpDfNoise = pd.DataFrame(tmpDfNoise, columns=['pixid', 'noisy', 'label'])
        dfNoise = dfNoise[['pixid']].merge(tmpDfNoise, 'right').combine_first(dfNoise).astype(dfNoise.dtypes)
        return dfNoise

    # @staticmethod
    # def generatorNoise(filename, noise_data_filename, noiseInfoFilename, noiseLevel, option):
    #     # Read data
    #     (nbClass, dates, classNames, samplesClass) = ReadGenerateData.readGenerateData(filename)
    #
    #     # Option = 1 : random noise per sample
    #     cpt = 1
    #     if option == 1:
    #         nbNoiseSamples = int(np.floor(noiseLevel[0, 0] * len(samplesClass)))  # Arrondi à l'inférieur
    #         noisyData = samplesClass.copy()
    #         for i in range(0, nbNoiseSamples):
    #             tabRandInt = np.random.randint(0, len(samplesClass), 2)
    #             noisyData[tabRandInt[0]] = [*samplesClass[tabRandInt[0], :2], *samplesClass[tabRandInt[1], 2:]]
    #             noisyData[tabRandInt[1]] = [*samplesClass[tabRandInt[1], :2], *samplesClass[tabRandInt[0], 2:]]

    # @staticmethod
    # def generatorNoisePerClass(filename, noise_data_filename, noiseInfoFilename, noiseLevel, option):
    #     # Read data
    #     (nbClass, dates, classNames, samplesClass) = ReadGenerateData.readGenerateData(filename)
    #
    #     # Option = 2 : random noise per sample
    #     cpt = 1
    #     if option == 2:
    #         tabClass = []
    #         tabShapeClass = []
    #         for i in range(1,len(classNames)+1):
    #             tabClass.append(samplesClass[samplesClass[:,0] == i])
    #             tabShapeClass.append(np.shape(tabClass[i-1])[0])
    #
    #         nbNoiseSamplesPerClass = np.floor(noiseLevel[0] * tabShapeClass)
    #         for i in range(0,len(tabClass)):
    #             tabClass[i] = np.random.permutation(tabClass[i])
    #             for j in range(0,int(nbNoiseSamplesPerClass[i])):
    #                 tmpRandClass = np.random.randint(0,len(tabClass))
    #                 while tmpRandClass == i:
    #                     tmpRandClass = np.random.randint(0, len(tabClass))
    #                 tmpRandSample = np.random.randint(0,tabShapeClass[i])
    #                 tabClass[i][j] = [tmpRandClass+1,*tabClass[i][tmpRandSample,1:]]
    #
    #         for j in range(0,len(tabClass)):
    #             nb = 0
    #             for i in tabClass[j][:,0]:
    #                 if i != j+1:
    #                     nb += 1
    #             print(str(np.round(nb/len(tabClass[j][:,0])*100))+'%')











