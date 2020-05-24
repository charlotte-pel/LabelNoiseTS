import hashlib
import math

from GenLabelNoiseTS.ReadGenerateData import *
import numpy as np


class GeneratorNoise:
    """

    """

    @staticmethod
    def generatorNoisePerClass(filename, rep, noiseLevel, seed, dfNbPixPerPolidList, dictClass=None, csv=False):
        """

        :param filename:
        :param noiseLevel:
        :param class1:
        :param class2:
        :return:
        """

        (nbClass, dates, classNames, samplesClass, nbPixelClass) = ReadGenerateData.readGenerateDataH5DataFrame(
            filename, rep, csv)

        # Number of pixels to be noisy
        nbNoiseSamplesPerClass = np.floor(np.multiply(noiseLevel, nbPixelClass))
        dfNoise = samplesClass[['pixid', 'label']]
        dfNoise.insert(1, "noisy", False, True)
        tmpDfNoise = []

        # Set random_state:
        randomState = np.random.RandomState(seed)

        def generateNoise(className, classNames, nbClassOneToMany=None):
            """
            Generate Noise on className with other class in classNames
            :param className: class to noisy
            :param classNames: other class
            :param nbClassOneToMany: In case: If One class to many: 'Wheat': ('Barley','Soy') nbClassOneToMany = 2 because ('Barley','Soy')
            :return: Nothing modify per reference tmpDfNoise
            """

            if nbClassOneToMany is None:
                nbNoiseSamples = int(nbNoiseSamplesPerClass[classNames.index(className)])
            else:
                nbNoiseSamples = nbClassOneToMany * nbPixPerPolid

            j = 0
            while j < nbNoiseSamples:
                tmpPixTab = np.array(
                    samplesClass.loc[
                        (samplesClass["label"] == className) & (samplesClass["polid"] == tmpIdTab[int(j / nbPixPerPolid)])][
                        'pixid'])
                newClass = randomState.choice(classNames, 1)[0]
                while newClass == className:
                    newClass = randomState.choice(classNames, 1)[0]
                for k in tmpPixTab:
                    if j < nbNoiseSamples:
                        tmpDfNoise.append((k, True, newClass))
                        j += 1

        # Systematic change
        if dictClass is not None:
            systematicChange = 'systematic_' + str(int(noiseLevel * 100)) + '_' + str(
                int(hashlib.md5(str(dictClass).encode("UTF-8")).hexdigest(), 16))
            for i in dictClass.items():
                # If One class to many: 'Wheat': ('Barley','Soy')
                if type(dictClass.get(i[0])) is tuple:
                    classNames = []
                    classNames.append(i[0])

                    # Polygon number distribution by noise class
                    # Ex: 'Wheat': ('Barley', 'Soy','Rapeseed') -> len(i[i]) = 3
                    nbPixPerPolid = int(np.array(dfNbPixPerPolidList[dfNbPixPerPolidList['label'] == classNames[0]]['nbPixelPerPolid']))
                    nbPolid = nbNoiseSamplesPerClass[classNames.index(classNames[0])] // nbPixPerPolid
                    nbPolidMod = int(nbPolid % len(i[1]))
                    nbPolid = nbPolid // len(i[1])
                    nbPolidList = np.ones((len(i[1]),), dtype=int) * nbPolid

                    for m in range(nbPolidMod + 1):
                        if m < nbPolidMod:
                            nbPolidList[m] += 1
                        elif m == nbPolidMod:
                            nbPolidList[m] += ((nbNoiseSamplesPerClass[classNames.index(classNames[0])] / nbPixPerPolid) % 1)
                    nbPolidList = randomState.permutation(nbPolidList)

                    tmpIdTab = randomState.permutation(
                        np.unique(np.array(samplesClass.loc[(samplesClass["label"] == classNames[0])]['polid'])))

                    for j, l in zip(i[1], range(len(i[1]))):
                        classNames.append(j)
                        generateNoise(classNames[0], classNames, nbPolidList[l])
                        del classNames[-1]
                        tmpIdTab = np.delete(tmpIdTab, slice(None, int(nbPolidList[l])))

                # Else One class to other class: 'Barley':'Soy'
                else:
                    classNames = i
                    tmpIdTab = randomState.permutation(
                        np.unique(np.array(samplesClass.loc[(samplesClass["label"] == classNames[0])]['polid'])))
                    generateNoise(classNames[0], classNames)
        # Random
        else:
            systematicChange = None
            for i in classNames:
                generateNoise(i, classNames)

        # print(np.array(tmpDfNoise).shape)
        # print(nbNoiseSamplesPerClass[0] * 2)
        tmpDfNoise = pd.DataFrame(tmpDfNoise, columns=['pixid', 'noisy', 'label'])
        for i in np.array(tmpDfNoise['pixid']):
            dfNoise = dfNoise[dfNoise['pixid'] != i]
        dfNoise = dfNoise.append(tmpDfNoise, ignore_index=True)
        dfNoise = dfNoise.sort_values(by=['pixid'])
        return noiseLevel, dfNoise, systematicChange
