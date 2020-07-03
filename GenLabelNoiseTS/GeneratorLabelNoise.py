import hashlib
from pathlib import Path
from GenLabelNoiseTS.ReadGenerateData import *


class GeneratorLabelNoise:
    """
    Noise generating class.
    """
    def __init__(self, filename, dir, noiseLevel, seed, dfNoPixPerPolidList, dictClass=None, csv=False):
        """

        :param filename: Name of the save file
        :param dir: Directory where the file is located
        :param noiseLevel: Percentage of label to be noisy
        :param seed: Seed for RandomState
        :param dfNoPixPerPolidList: Dataframe containg a list of number of pixel per polygon
        :param dictClass: The dictionary used for systematic change
        :param csv: If the option csv is activated
        """
        self._filename = filename
        self._dir = Path(dir)
        self._csv = csv
        self._dictClass = dictClass
        (_, _, self._classNames, self._samplesClass,
         self._noPixelClass) = ReadGenerateData.readGenerateDataH5DataFrame(
            self._filename, self._dir, self._csv)
        self._noiseLevel = noiseLevel

        # Number of pixels to be noisy
        self._noNoiseSamplesPerClass = np.floor(np.multiply(self._noiseLevel, self._noPixelClass))
        self._dfNoise = self._samplesClass[['pixid', 'label']]
        self._dfNoise.insert(1, "noisy", False, True)
        self._tmpDfNoise = []
        self._seed = seed
        self._dfNoPixPerPolidList = dfNoPixPerPolidList

        # Set random_state:
        self._randomState = np.random.RandomState(self._seed)

    def generatorNoisePerClass(self):
        """

        :return: self._noiseLevel, self._dfNoise, systematicChange
        """

        # Systematic change
        if self._dictClass is not None:
            systematicChange = 'systematic_' + str(int(self._noiseLevel * 100)) + '_' + str(
                int(hashlib.md5(str(self._dictClass).encode("UTF-8")).hexdigest(), 16))
            for i in self._dictClass.items():
                # If One class to many: 'Wheat': ('Barley','Soy')
                if type(self._dictClass.get(i[0])) is tuple:

                    noNoisyClass = len(i[1])
                    classNames = []
                    classNames.append(i[0])
                    (noPixPerPolid, noPolidMod, noPolidList) = self._getNoPixPerPolid(noNoisyClass,classNames[0])

                    for m in range(noPolidMod + 1):
                        if m < noPolidMod:
                            noPolidList[m] += 1
                        elif m == noPolidMod:
                            noPolidList[m] += ((self._noNoiseSamplesPerClass[classNames[0]] / noPixPerPolid) % 1)

                    # If use math.ceil
                    # noPolidList = self._randomState.permutation(noPolidList)

                    tmpPolidTab = self._randomState.permutation(
                        np.unique(
                            np.array(self._samplesClass.loc[(self._samplesClass["label"] == classNames[0])]['polid'])))
                    for j, l in zip(self._randomState.permutation(np.array(i[1])), range(len(i[1]))):
                        classNames.append(j)
                        self._generateNoise(classNames[0], classNames, tmpPolidTab, noPixPerPolid, noPolidList[l])
                        del classNames[-1]
                        tmpPolidTab = np.delete(tmpPolidTab, slice(None, int(noPolidList[l])))
                        # tmpPolidTab = np.delete(tmpPolidTab, slice(None, math.ceil(noPolidList[l])))

                # Else One class to other class: 'Barley':'Soy'
                else:
                    classNames = i
                    noPixPerPolid = int(
                        np.array(self._dfNoPixPerPolidList[self._dfNoPixPerPolidList['label'] == classNames[0]][
                                     'noPixelPerPolid']))
                    tmpPolidTab = self._randomState.permutation(
                        np.unique(
                            np.array(self._samplesClass.loc[(self._samplesClass["label"] == classNames[0])]['polid'])))
                    self._generateNoise(classNames[0], classNames, tmpPolidTab, noPixPerPolid)

        # Random
        else:
            systematicChange = None
            for i in self._classNames:
                noPixPerPolid = int(
                    np.array(self._dfNoPixPerPolidList[self._dfNoPixPerPolidList['label'] == i][
                                 'noPixelPerPolid']))
                tmpPolidTab = self._randomState.permutation(
                    np.unique(
                        np.array(self._samplesClass.loc[(self._samplesClass["label"] == i)]['polid'])))
                self._generateNoise(i, self._classNames, tmpPolidTab, noPixPerPolid)

        # Creation of Noisy Dataframe
        self._tmpDfNoise = pd.DataFrame(self._tmpDfNoise, columns=['pixid', 'noisy', 'label'])
        for i in np.array(self._tmpDfNoise['pixid']):
            self._dfNoise = self._dfNoise[self._dfNoise['pixid'] != i]
        self._dfNoise = self._dfNoise.append(self._tmpDfNoise, ignore_index=True)
        self._dfNoise = self._dfNoise.sort_values(by=['pixid'])

        return self._noiseLevel, self._dfNoise, systematicChange

    def _generateNoise(self, className, classNames, tmpPolidTab, noPixPerPolid, noPolid=None):
        """
        Generate Noise on className with other class in classNames
        :param className: class to noisy
        :param classNames: other class
        :param tmpPolidTab: Array containing list of polid for "className"
        :param noPixPerPolid: Number of pixel per polygon
        :param noPolid: Number of polygon
        :return: Nothing modify per reference self._tmpDfNoise
        """
        if noPolid is None:
            noNoiseSamples = int(self._noNoiseSamplesPerClass[className])
        else:
            noNoiseSamples = noPolid * noPixPerPolid
        j = 0
        while j < noNoiseSamples:
            tmpPixTab = np.array(
                self._samplesClass.loc[
                    (self._samplesClass["label"] == className) & (
                                self._samplesClass["polid"] == tmpPolidTab[int(j / noPixPerPolid)])][
                    'pixid'])
            newClass = self._randomState.choice(classNames, 1)[0]
            while newClass == className:
                newClass = self._randomState.choice(classNames, 1)[0]
            for k in tmpPixTab:
                if j < noNoiseSamples:
                    self._tmpDfNoise.append((k, True, newClass))
                    j += 1

    def _getNoPixPerPolid(self, noNoisyClass, className):
        """

        :param itemDict: Name of the class
        :param classNames: Array containing Class Names
        :return: noPixPerPolid, noPolidMod, noPolidList
        """
        # Polygon number distribution by noise class
        # Ex: 'Wheat': ('Barley', 'Soy','Rapeseed') -> len(i[i]) = 3
        noPixPerPolid = int(
            np.array(self._dfNoPixPerPolidList[self._dfNoPixPerPolidList['label'] == className][
                         'noPixelPerPolid']))
        noPolid = np.array(self._noNoiseSamplesPerClass[className]) // noPixPerPolid
        noPolidMod = int(noPolid % noNoisyClass)
        noPolid = noPolid // noNoisyClass
        noPolidList = np.ones((noNoisyClass,), dtype=int) * noPolid

        return noPixPerPolid, noPolidMod, noPolidList
