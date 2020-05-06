import pandas as pd


class ReadGenerateData:

    @staticmethod
    def readGenerateDataH5DataFrame(filename):
        dfheader = pd.read_hdf(filename, 'header')
        dfData = pd.read_hdf(filename, 'data')
        dfheader = pd.DataFrame(dfheader)
        dfData = pd.DataFrame(dfData)
        nbClass = len(dfheader) - 1
        dates = dfheader[0][0]
        classNames = []
        for i in range(1, nbClass + 1):
            classNames.append(dfheader[0][i][0])
        samplesClass = dfData
        nbPixelClass = [len(samplesClass.loc[(samplesClass['label'] == i)]) for i in classNames]
        # Doesn't work in all cases: if size of poly aren't the same. The size is fixed -> samplesClass['polid'] == 1
        nbPolyClass = [len(samplesClass.loc[(samplesClass['label'] == i)]) / len(
            samplesClass.loc[(samplesClass['label'] == i) & (samplesClass['polid'] == 1)]) for i in classNames]
        return nbClass, dates, classNames, samplesClass, nbPixelClass, nbPolyClass

    # @staticmethod
    # def readGenerateData(filename):
    #     """
    #
    #     :param filename: filename: the name of the file in which the data will be entered
    #     :return: nb_class: nb_class: number of class
    #              dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
    #              class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
    #              samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20] (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
    #     """
    #     # Test Filename
    #     if not os.path.exists(filename):
    #         print('Invalid filename: ' + filename)
    #
    #     arrayFid = []
    #     fid = open(filename, "r")
    #
    #     # Format each line of file
    #     for line in fid:
    #         arrayFid.append(line.replace("\n", ""))
    #
    #     # Read header
    #     nb_class = int(arrayFid[0])
    #     dates = arrayFid[1].split(';')
    #     del dates[-1]
    #     for i in range(0, len(dates)):
    #         dates[i] = int(dates[i])
    #
    #     # Class names list
    #     class_names = []
    #     for i in range(2, nb_class + 2):
    #         class_names.append(arrayFid[i].split(";")[1])
    #
    #     # Store samples per class
    #     cpt = 0
    #     samplesClass = -np.ones((len(arrayFid) - 15, 17))
    #     for i in arrayFid:
    #         if cpt >= 2 + nb_class:
    #             tmp = i.split(";")
    #             k = 0
    #             del tmp[-1]
    #             for j in tmp:
    #                 samplesClass[cpt - 15, k] = j.replace("c", "").replace("id", "")
    #                 k += 1
    #         cpt += 1
    #     fid.close()
    #     return nb_class, dates, class_names, samplesClass
    #
    # @staticmethod
    # def readGenerateDataH5(filename):
    #     with h5py.File(filename, 'r') as fid:
    #         classNames = list(fid.keys())
    #         dates = np.array(fid.get(classNames[-1]))
    #         del classNames[-1]
    #         nbClass = len(classNames)
    #         samplesClass = []
    #         for i in classNames:
    #             for j in np.array(fid.get(i)):
    #                 id = fid.get(i + '/' + j)
    #                 for k in id:
    #                     samplesClass.append([int(classNames.index(i) + 1), int(j), *k[0]])
    #
    #         samplesClass = np.array(samplesClass)  # ,dtype=dType)
    #
    #         return nbClass, dates, classNames, samplesClass
    #
    # @staticmethod
    # def getSpeIdSpeClass(filename, className, IdPoly):
    #     with h5py.File(filename, 'r') as fid:
    #         classNames = list(fid.keys())
    #         del classNames[-1]
    #         idPolyTab = np.array(fid.get(className + '/' + str(IdPoly)))
    #         tabIdPoly = []
    #         for i in idPolyTab:
    #             tabIdPoly.append([int(classNames.index(className) + 1), IdPoly, *i[0]])
    #         tabIdPoly = np.array(tabIdPoly)
    #         return tabIdPoly
    #
    # @staticmethod
    # def getSpeClass(filename, className):
    #     """
    #
    #     :param filename:
    #     :param className:
    #     :return: Matrix with shape(50, 9, 17) -> 50 = nbPoly, 9 = nbPixel, 17 = nbParam
    #     """
    #     with h5py.File(filename, 'r') as fid:
    #         classNames = list(fid.keys())
    #         del classNames[-1]
    #         idPolyTab = np.array(fid.get(className))
    #         nbPixel = len(np.array(fid.get(className + '/' + idPolyTab[0])))
    #         sizePixel = len(np.array(fid.get(className + '/' + idPolyTab[0]))[0,0])
    #         tabClass = np.empty(shape=(len(idPolyTab), nbPixel,sizePixel+2))
    #         #print(tabClass.shape)
    #         j = 0
    #         for i in idPolyTab:
    #             id = fid.get(className + '/' + i)
    #             l = 0
    #             for k in id:
    #                 tabClass[j,l] = [int(classNames.index(className) + 1), int(i), *k[0]]
    #                 l += 1
    #             j += 1
    #         return tabClass

    # @staticmethod
    # def readGenerateDataH5DataFrame(filename):
    #     with h5py.File(filename, 'r') as fid:
    #         classNames = list(fid.keys())
    #         dates = np.array(fid.get(classNames[-1]))
    #         del classNames[-1]
    #         nbClass = len(classNames)
    #         tmpTab = []
    #         df = pd.DataFrame(columns=['label','polid','pixid','profil'])
    #         l = 0
    #         for i in classNames:
    #              for j in np.array(fid.get(i)):
    #                  id = fid.get(i + '/' + j)
    #                  g = 0
    #                  for k in id:
    #                     df2 = pd.DataFrame(np.array([[i,int(j),g,k[0]]]),columns=['label','polid','pixid','profil'])
    #                     df = df.append(df2, ignore_index = True)
    #                     g += 1
    #                     l += 1
    #
    #         df.to_hdf('dataFrame.h5', key='df', mode='w')
    #         #return nbClass, dates, classNames, samplesClass
