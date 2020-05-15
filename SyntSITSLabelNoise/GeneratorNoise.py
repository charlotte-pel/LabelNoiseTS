import hashlib

from SyntSITSLabelNoise.ReadGenerateData import *
import numpy as np

class GeneratorNoise:
    """

    """

    @staticmethod
    def generatorNoisePerClass(filename,rep, noiseLevel, dictClass=None,csv=False,seed=0):
        """

        :param filename:
        :param noiseLevel:
        :param class1:
        :param class2:
        :return:
        """

        (nbClass, dates, classNames, samplesClass, nbPixelClass) = ReadGenerateData.readGenerateDataH5DataFrame(filename,rep,csv)
        nbNoiseSamplesPerClass = np.floor(np.multiply(noiseLevel, nbPixelClass))
        dfNoise = samplesClass[['pixid', 'label']]
        dfNoise.insert(1, "noisy", False, True)
        tmpDfNoise = []

        def generateNoise(className, classNames, nbClassOneToMany=None):
            """
            Generate Noise on className with other class in classNames
            :param className: class to noisy
            :param classNames: other class
            :param nbClassOneToMany: In case: If One class to many: 'Wheat': ('Barley','Soy') nbClassOneToMany = 2 because ('Barley','Soy')
            :return: Nothing modify per reference tmpDfNoise
            """
            # Set random_state:
            randomState = np.random.RandomState(seed)

            tmpIdTab = randomState.permutation(
                np.unique(np.array(samplesClass.loc[(samplesClass["label"] == className)]['polid'])))
            j = 0
            if nbClassOneToMany is None:
                nbNoiseSamples = int(nbNoiseSamplesPerClass[classNames.index(className)])
            else:
                nbNoiseSamples = int(nbNoiseSamplesPerClass[classNames.index(className)]/nbClassOneToMany)

            while j < nbNoiseSamples:
                tmpPixTab = np.array(
                    samplesClass.loc[(samplesClass["label"] == className) & (samplesClass["polid"] == tmpIdTab[int(j / 10)])][
                        'pixid'])
                newClass = randomState.choice(classNames, 1)[0]
                while newClass == className:
                    newClass = randomState.choice(classNames,1)[0]
                for k in tmpPixTab:
                    if j < nbNoiseSamples:
                        tmpDfNoise.append((k, True, newClass))
                        j += 1

        # Systematic change
        if dictClass is not None:
            systematicChange = 'systematic_'+str(int(noiseLevel * 100))+'_'+str(int(hashlib.md5(str(dictClass).encode("UTF-8")).hexdigest(), 16))
            for i in dictClass.items():
                # If One class to many: 'Wheat': ('Barley','Soy')
                if type(dictClass.get(i[0])) is tuple:
                    classNames = []
                    classNames.append(i[0])
                    for j in i[1]:
                        classNames.append(j)
                        generateNoise(classNames[0], classNames, len(i[1]))
                        del classNames[-1]
                    classNames = i
                # Else One class to other class: 'Barley':'Soy'
                else:
                    classNames = i
                    generateNoise(classNames[0], classNames)
        # Random
        else:
            systematicChange = None
            for i in classNames:
                generateNoise(i,classNames)

        #print(np.array(tmpDfNoise).shape)
        #print(nbNoiseSamplesPerClass[0] * 2)
        tmpDfNoise = pd.DataFrame(tmpDfNoise, columns=['pixid', 'noisy', 'label'])
        dfNoise = dfNoise[['pixid']].merge(tmpDfNoise, 'right').combine_first(dfNoise).astype(dfNoise.dtypes)
        return noiseLevel, dfNoise, systematicChange











