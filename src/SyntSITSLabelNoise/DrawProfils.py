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
        index = class_names.index(class_name) + 1 # L'index dans le fichier commence à 1 alors que python commence à 0
        cpt = 0
        for i in samplesClass:
            if i[0] == index:
                cpt += 1
        print(cpt)
        samplesSpeClass = -np.ones((cpt, 15))
        k = 0
        for i in samplesClass:
            l = 0
            #print(i)
            if i[0] == index:
                for j in i:
                    if l >= 2:
                        samplesSpeClass[k, l - 2] = j
                    l += 1
                k += 1

        for i in samplesSpeClass:
            plt.plot(dates, i)
        plt.title('Profils NDVI simulés '+class_name)
        plt.grid()
        plt.xlabel('Jour de l\'an')
        plt.ylabel('NDVI')
        plt.axis([0, 350, 0, 1])
        plt.show()
