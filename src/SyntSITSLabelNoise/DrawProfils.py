import matplotlib.pyplot as plt
import numpy as np


class Drawprofils:

    @staticmethod
    def drawProfilClass(class_name, nb_class, dates, class_names, samplesClass):
        """

        :param class_name:
        :param nb_class: number of class
        :param dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
        :param class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
        :param samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
               (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        :return: No return -> Draw graph
        """
        index = class_names.index(class_name) + 1  # L'index dans le fichier commence à 1 alors que python commence à 0
        cpt = 0
        for i in samplesClass:
            if i[0] == index:
                cpt += 1
        print(cpt)
        samplesSpeClass = -np.ones((cpt, 15))
        k = 0
        for i in samplesClass:
            l = 0
            # print(i)
            if i[0] == index:
                for j in i:
                    if l >= 2:
                        samplesSpeClass[k, l - 2] = j
                    l += 1
                k += 1
        for i in samplesSpeClass:
            plt.plot(dates, i)
        plt.title('Profils NDVI simulés ' + class_name)
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        plt.show()

    @staticmethod
    def drawProfilMeanClass(nb_class, dates, class_names, samplesClass):
        """

        :param class_name:
        :param nb_class: number of class
        :param dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
        :param class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
        :param samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
               (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        :return: No return -> Draw graph
        """
        for i in range(0, nb_class - 2):
            Drawprofils.drawMeanOneClass(i, dates, class_names, samplesClass)
        plt.title('Profils NDVI simulés: Mean Each Class ')
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0.2, 1])
        plt.show()

    @staticmethod
    def drawProfilMeanSpeClass(ids, nb_class, dates, class_names, samplesClass,options):
        """

        :param class_name:
        :param nb_class: number of class
        :param dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
        :param class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
        :param samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
               (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        :return: No return -> Draw graph
        """
        nameOfClass = ''
        for i in ids:
            Drawprofils.drawMeanOneClass(i, dates, class_names, samplesClass,options)
            nameOfClass = nameOfClass + ' / ' + str(class_names[i])
        if len(ids) <= 4:
            plt.title('Profils NDVI simulés: ' + nameOfClass)
        else:
            plt.title('Profils NDVI simulés')
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0.2, 0.9])
        plt.show()

    @staticmethod
    def drawMeanOneClass(id, dates, class_names, samplesClass,option):
        index = id + 1
        cpt = 0
        for i in samplesClass:
            if int(i[0]) == index:
                cpt += 1
        print(cpt)
        samplesSpeClass = -np.ones((cpt, 15))
        k = 0
        for i in samplesClass:
            l = 0
            if int(i[0]) == index:
                if option == 0:
                    for j in i:
                        if l >= 2:
                            samplesSpeClass[k, l - 2] = j
                        l += 1
                    k += 1
                else:
                    samplesSpeClass[k] = i[2:]
                    k += 1

        plt.plot(dates, np.mean(samplesSpeClass, axis=0), label=class_names[index - 1])

    @staticmethod
    def drawMeanProfilOneClass(id, dates, class_names, samplesClass):
        """
        Draw Mean per Profil for OneClass
        :param id:
        :param dates:
        :param class_names:
        :param samplesClass:
        :return:
        """
        index = id + 1
        cpt = 0
        nbId = 0
        for i in samplesClass:
            if i[0] == index:
                cpt += 1
                nbId = i[1]
        print(cpt)
        print(nbId)
        samplesSpeClass = -np.ones((cpt, 16))
        k = 0
        for i in samplesClass:
            l = 0
            if i[0] == index:
                for j in i:
                    if l >= 1:
                        samplesSpeClass[k, l - 1] = j
                    l += 1
                k += 1
        samplesMean = -np.ones((int(nbId), 16))
        # A[A[:,1] == i]
        for i in range(1,int(nbId)):
            samples = samplesSpeClass[samplesSpeClass[:, 0] == i]
            samplesMean[i] = np.mean(samples, axis=0)
        print(samplesMean[:,1:])
        for i in samplesMean:
            plt.plot(dates, i[1:], label=class_names[index - 1])
        plt.title('Profils NDVI simulés ' + class_names[id])
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0.2, 1])
        plt.grid()
        plt.show()

    @staticmethod
    def draw20RandomProfilOneClass(id, dates, class_names, samplesClass):
        index = id+1
        cpt = 0
        nbId = 0
        for i in samplesClass:
            if i[0] == index:
                cpt += 1
                nbId = i[1]
        print(cpt)
        samplesSpeClass = -np.ones((cpt, 15))
        k = 0
        for i in samplesClass:
            l = 0
            # print(i)
            if i[0] == index:
                for j in i:
                    if l >= 2:
                        samplesSpeClass[k, l - 2] = j
                    l += 1
                k += 1
        for j in range(0,20):
            i = np.random.randint(0,nbId)
            plt.plot(dates, samplesSpeClass[i])
        plt.title('Profils NDVI simulés ' + class_names[id])
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        plt.show()

    @staticmethod
    def draw20RandomIdProfilOneClass(id, dates, class_names, samplesClass):
        index = id + 1
        cpt = 0
        nbId = 0
        for i in samplesClass:
            if i[0] == index:
                cpt += 1
                nbId = i[1]
        print(cpt)
        print(nbId)
        samplesSpeClass = -np.ones((cpt, 16))
        k = 0
        for i in samplesClass:
            l = 0
            if i[0] == index:
                for j in i:
                    if l >= 1:
                        samplesSpeClass[k, l - 1] = j
                    l += 1
                k += 1
        i = np.random.randint(1, nbId)
        samples = samplesSpeClass[samplesSpeClass[:, 0] == i]
        print(i)
        for j in range(0, 20):
            if j < len(samples):
                tmpSample = samples[j]
                plt.plot(dates, tmpSample[1:])
        plt.title('Profils NDVI simulés ' + class_names[id])
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        plt.show()
