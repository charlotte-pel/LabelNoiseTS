import hashlib
import math

from GenLabelNoiseTS.ReadGenerateData import *
import numpy as np


class GeneratorNoise:
    """

    """

    @staticmethod
    def generatorNoisePerClass(filename, rep, noiseLevel, seed, dictClass=None, csv=False):
        """

        :param filename:
        :param noiseLevel:
        :param class1:
        :param class2:
        :return:
        """

        (nbClass, dates, classNames, samplesClass, nbPixelClass) = ReadGenerateData.readGenerateDataH5DataFrame(
            filename, rep, csv)
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
                nbNoiseSamples = nbClassOneToMany * 10
                print(nbNoiseSamples)

            j = 0
            while j < nbNoiseSamples:
                tmpPixTab = np.array(
                    samplesClass.loc[
                        (samplesClass["label"] == className) & (samplesClass["polid"] == tmpIdTab[int(j / 10)])][
                        'pixid'])
                newClass = randomState.choice(classNames, 1)[0]
                while newClass == className:
                    newClass = randomState.choice(classNames, 1)[0]
                for k in tmpPixTab:
                    if j < nbNoiseSamples:
                        tmpDfNoise.append((k, True, newClass))
                        j += 1
            return int(nbNoiseSamples / 10)

        # Systematic change
        if dictClass is not None:
            systematicChange = 'systematic_' + str(int(noiseLevel * 100)) + '_' + str(
                int(hashlib.md5(str(dictClass).encode("UTF-8")).hexdigest(), 16))
            for i in dictClass.items():
                # If One class to many: 'Wheat': ('Barley','Soy')
                if type(dictClass.get(i[0])) is tuple:
                    classNames = []
                    classNames.append(i[0])

                    cal = nbNoiseSamplesPerClass[classNames.index(classNames[0])] // 10
                    calMod = int(cal % len(i[1]))
                    cal = cal // len(i[1])
                    calList = np.ones((len(i[1]),), dtype=int) * cal
                    for m in range(calMod + 1):
                        if m < calMod:
                            calList[m] += 1
                        elif m == calMod:
                            calList[m] += ((nbNoiseSamplesPerClass[classNames.index(classNames[0])] / 10) % 1)
                    calList = randomState.permutation(calList)
                    print(calList)

                    tmpIdTab = randomState.permutation(
                        np.unique(np.array(samplesClass.loc[(samplesClass["label"] == classNames[0])]['polid'])))
                    for j, l in zip(i[1], range(len(i[1]))):
                        classNames.append(j)
                        k = generateNoise(classNames[0], classNames, calList[l])
                        del classNames[-1]
                        tmpIdTab = np.delete(tmpIdTab, slice(None, k))
                        # print(len(tmpIdTab))
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
