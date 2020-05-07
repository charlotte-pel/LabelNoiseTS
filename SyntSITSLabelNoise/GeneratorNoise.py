from SyntSITSLabelNoise.ReadGenerateData import *

class GeneratorNoise:
    """

    """

    @staticmethod
    def generatorNoisePerClass(filename, noiseLevel, dictClass=None):
        """

        :param filename:
        :param noiseLevel:
        :param class1:
        :param class2:
        :return:
        """

        (nbClass, dates, classNames, samplesClass, nbPixelClass) = ReadGenerateData.readGenerateDataH5DataFrame(filename)
        nbNoiseSamplesPerClass = np.floor(np.multiply(noiseLevel, nbPixelClass))
        dfNoise = samplesClass[['pixid', 'label']]
        dfNoise.insert(1, "noisy", False, True)
        tmpDfNoise = []

        def generateNoise(className, classNames):
            """
            Generate Noise on className with other class in classNames
            :param className: class to noisy
            :param classNames: other class
            :return: Nothing modify per reference tmpDfNoise
            """
            tmpIdTab = np.random.permutation(
                np.unique(np.array(samplesClass.loc[(samplesClass["label"] == className)]['polid'])))
            j = 0
            nbNoiseSamples = int(nbNoiseSamplesPerClass[classNames.index(className)])
            while j < nbNoiseSamples:
                tmpPixTab = np.array(
                    samplesClass.loc[(samplesClass["label"] == className) & (samplesClass["polid"] == tmpIdTab[int(j / 10)])][
                        'pixid'])
                newClass = np.random.choice(classNames, 1)
                while newClass is className:
                    newClass = np.random.choice(classNames)
                newClass = newClass[0]
                for k in tmpPixTab:
                    if j < nbNoiseSamples:
                        tmpDfNoise.append((k, True, newClass))
                        j += 1

        if dictClass is not None:
            systematicChange = ''
            for i in dictClass.items():
                classNames = i
                systematicChange = systematicChange+classNames[0]+'To'+classNames[1]+'_'
                generateNoise(classNames[0],classNames)
        else:
            systematicChange = None
            for i in classNames:
                generateNoise(i,classNames)

        print(np.array(tmpDfNoise).shape)
        print(nbNoiseSamplesPerClass[0] * nbClass)
        tmpDfNoise = pd.DataFrame(tmpDfNoise, columns=['pixid', 'noisy', 'label'])
        dfNoise = dfNoise[['pixid']].merge(tmpDfNoise, 'right').combine_first(dfNoise).astype(dfNoise.dtypes)
        return noiseLevel, dfNoise, systematicChange












