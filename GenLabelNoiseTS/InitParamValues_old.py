import numpy as np
import pandas as pd


class InitParamValues:

    @staticmethod
    def generateDatesComplete(start='1/1/2018', freqDays=25, periods=15):
        """

        :param start: first date
        :param freqDays: time between two dates
        :param periods: nb dates = 15 dates in array
        :return: array with complete dates: 1/1/2018
        """
        dateArray = pd.date_range(start=start, freq=str(freqDays) + 'D', periods=periods)
        return dateArray

    @staticmethod
    def generateDates():
        """

        :return: array containing days spaced 25 days apart
        """
        return [i for i in np.arange(1,365,25)]

    @staticmethod
    def initParamValues(nb_params=26, nb_class_panel=13):
        """

        :param nb_params: number of param, by default = 26
        :param nb_class_panel: number of class, by default = 14
        :return: param_val: param of double sigmo for each class
                 param_val[:14, 0] = [0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
        """

        # Class_names
        class_names = ['Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy',
                       'Wheat', 'Rapeseed', 'Barley',
                       'Wheat_Soy',
                       'Evergreen', 'Decideous',
                       'Water',
                       'Build',
                       'other']

        param_val = -np.ones((nb_params, nb_class_panel))
        # Corn t0= [90 155] t1 = [185 210]
        param_val[:14, 0] = [0.57, 0.72, 0.15, 0.3, 100, 200, 5, 25, 250, 310, 10, 30, 500, 10]
        # Corn ensilage t0= [90 155] t1 = [185 210]
        param_val[:14, 1] = [0.57, 0.72, 0.15, 0.3, 100, 200, 5, 25, 250, 310, 5, 10, 500, 10]
        # Sorghum t0= [150 195] t1 = [195 220]
        param_val[:14, 2] = [0.62, 0.77, 0.15, 0.30, 120, 190, 20, 40, 290, 295, 25, 30, 500, 10]
        # Sunflower
        param_val[:14, 3] = [0.67, 0.82, 0.15, 0.30, 102, 192, 15, 40, 180, 240, 5, 20, 500, 10]
        # Soy peak in mid-september (doy=260)
        param_val[:14, 4] = [0.67, 0.82, 0.15, 0.30, 140, 220, 15, 45, 270, 320, 20, 45, 500, 10]
        # Wheat
        param_val[:14, 5] = [0.52, 0.67, 0.20, 0.35, 30, 90, 5, 25, 125, 175, 5, 25, 500, 10]
        # Rapeseed 0.5500    0.1000   40.0000   20.0000   85.0000    5.0000    0.4500    0.1000   90.0000    5.0000
        # 140.0000   10.0000
        param_val[:, 6] = [0.70, 0.8, 0.05, 0.20, 30, 45, 15, 25, 80, 90, 3, 12, 500, 10, 0.60, 0.70, 0.05, 0.15,
                            85,
                            95, 3, 12, 135, 145, 5, 15]
        # Barley
        param_val[:14, 7] = [0.52, 0.67, 0.20, 0.35, 30, 90, 5, 25, 120, 170, 5, 25, 500, 10]
        # Bi-culture: wheat-soy
        param_val[:, 8] = [0.5, 0.55, 0.10, 0.15, 45, 55, 10, 15, 130, 135, 2, 5, 500, 10, 0.65, 0.7, 0.10, 0.15,
                           170,
                           180, 10, 15, 280, 300, 25, 35]
        # Evergreen
        param_val[:14, 9] = [0.01, 0.015, 0.55, 0.7, 0, 365, 100, 150, 0, 365, 100, 150, 500, 10]
        # Decideous
        param_val[:14, 10] = [0.2, 0.35, 0.40, 0.5, 23, 27, 15, 20, 315, 320, 15, 20, 500, 10]
        # Water
        param_val[:14, 11] = [0.01, 0.02, -0.2, 0, 150, 200, 10, 15, 50, 200, 10, 15, 500, 10]
        # Build
        param_val[:14, 12] = [0.01, 0.02, 0.2, 0.3, 0, 365, 10, 15, 0, 365, 10, 15, 500, 10]

        dfTest = pd.DataFrame(np.transpose(param_val))
        del class_names[-1]
        dfTest.insert(0,'class_names',class_names,True)
        test = pd.DataFrame([InitParamValues.generateDates(),])
        dfTest = dfTest.append(test,ignore_index=True)
        #dfTest.to_csv("initFile.csv", index=False)
        param_val = dfTest

        return param_val, class_names