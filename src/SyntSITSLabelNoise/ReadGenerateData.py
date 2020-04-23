import os

import numpy as np


class ReadGenerateData:

    @staticmethod
    def read_generate_data(filename):
        """

        :param filename: filename: the name of the file in which the data will be entered
        :return: nb_class: nb_class: number of class
                 dates: dates: number of days since New Year's Day, dates = [0,25,50,...]
                 class_names: class_names: contains the names of different class, class_names = ['Corn', 'Corn_ensilage',...]
                 samplesClass: each line are -> [idClass,idnb,0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20] (idClass -> int: 1,2,.. ; idnb -> int: 1,2,..)
        """
        # Test Filename
        if not os.path.exists(filename):
            print('Invalid filename: ' + filename)

        arrayFid = []
        fid = open(filename, "r")

        # Format each line of file
        for line in fid:
            arrayFid.append(line.replace("\n", ""))

        # Read header
        nb_class = int(arrayFid[0])
        dates = arrayFid[1].split(';')
        del dates[-1]
        for i in range(0, len(dates)):
            dates[i] = int(dates[i])

        # Class names list
        class_names = []
        for i in range(2, nb_class + 2):
            class_names.append(arrayFid[i].split(";")[1])

        # Store samples per class
        cpt = 0
        samplesClass = -np.ones((len(arrayFid) - 15, 17))
        for i in arrayFid:
            if cpt >= 2 + nb_class:
                tmp = i.split(";")
                k = 0
                del tmp[-1]
                for j in tmp:
                    samplesClass[cpt - 15, k] = j.replace("c", "").replace("id", "")
                    k += 1
            cpt += 1
        fid.close()
        return nb_class, dates, class_names, samplesClass
