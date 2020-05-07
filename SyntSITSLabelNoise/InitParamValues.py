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
    def initParamValues(option, nb_params=26, nb_class_panel=13):
        """

        :param option: three options are available: 1,2 or 3
        :param nb_params: number of param, by default = 26
        :param nb_class_panel: number of class, by default = 14
        :return: param_val: param of double sigmo for each class
                 param_val[:14, 0] = [0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
        """
        if option < 3:
            option = 1

        # Class_names
        class_names = ['Corn', 'Corn_ensilage', 'Sorghum', 'Sunflower', 'Soy',
                       'Wheat', 'Rapeseed', 'Barley',
                       'Wheat_soy',
                       'Evergreen', 'Decideous',
                       'Water',
                       'Build',
                       'other']
        param_val = -np.ones((nb_params, nb_class_panel))
        if option == 1:
            # True profile
            # Corn t0= [90 155] t1 = [185 210]
            param_val[:14, 0] = [0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
            # Corn ensilage t0= [90 155] t1 = [185 210]
            param_val[:14, 1] = [0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 260, 280, 5, 10, 500, 20]
            # Sorghum t0= [150 195] t1 = [195 220]
            param_val[:14, 2] = [0.67, 0.72, 0.2, 0.25, 140, 170, 25, 35, 290, 295, 25, 30, 500, 20]
            # Sunflower
            param_val[:14, 3] = [0.72, 0.77, 0.2, 0.25, 122, 172, 20, 35, 200, 220, 10, 15, 500, 20]
            # Soy peak in mid-september (doy=260)
            param_val[:14, 4] = [0.72, 0.77, 0.2, 0.25, 160, 180, 25, 35, 290, 300, 30, 35, 500, 20]
            # Wheat
            param_val[:14, 5] = [0.57, 0.62, 0.25, 0.3, 50, 70, 10, 15, 145, 155, 10, 15, 500, 20]
            # Rapeseed 0.5500    0.1000   40.0000   20.0000   85.0000    5.0000    0.4500    0.1000   90.0000    5.0000
            # 140.0000   10.0000
            param_val[:, 6] = [0.73, 0.78, 0.1, 0.15, 35, 40, 18, 22, 83, 87, 3, 7, 500, 20, 0.62, 0.67, 0.07, 0.13, 88,
                               92,
                               3, 7, 137, 143, 7, 13]
            # Barley
            param_val[:14, 7] = [0.57, 0.62, 0.25, 0.3, 50, 70, 10, 15, 140, 150, 10, 15, 500, 20]
            # Bi-culture: wheat-soy
            param_val[:, 8] = [0.5, 0.55, 0.10, 0.15, 45, 55, 10, 15, 130, 135, 2, 5, 500, 20, 0.65, 0.7, 0.10, 0.15,
                               170,
                               180, 10, 15, 280, 300, 25, 35]
            # Evergreen
            param_val[:14, 9] = [0.01, 0.015, 0.65, 0.7, 0, 365, 100, 150, 0, 365, 100, 150, 500, 20]
            # Decideous
            param_val[:14, 10] = [0.3, 0.35, 0.45, 0.5, 23, 27, 15, 20, 315, 320, 15, 20, 500, 20]
            # Water
            param_val[:14, 11] = [0.01, 0.02, -0.2, 0, 150, 200, 10, 15, 50, 200, 10, 15, 500, 20]
            # Build
            param_val[:14, 12] = [0.01, 0.02, 0.2, 0.3, 0, 365, 10, 15, 0, 365, 10, 15, 500, 20]

        elif option == 2:

            # Corn t0= [90 155] t1 = [185 210]
            param_val[:14, 0] = [0.57, 0.72, 0.15, 0.3, 100, 200, 5, 25, 250, 310, 10, 30, 500, 20]
            # Corn ensilage t0= [90 155] t1 = [185 210]
            param_val[:14, 1] = [0.57, 0.72, 0.15, 0.3, 100, 200, 5, 25, 250, 310, 5, 10, 500, 20]
            # Sorghum t0= [150 195] t1 = [195 220]
            param_val[:14, 2] = [0.62, 0.77, 0.15, 0.30, 120, 190, 20, 40, 290, 295, 25, 30, 500, 20]
            # Sunflower
            param_val[:14, 3] = [0.67, 0.82, 0.15, 0.30, 102, 192, 15, 40, 180, 240, 5, 20, 500, 20]
            # Soy peak in mid-september (doy=260)
            param_val[:14, 4] = [0.67, 0.82, 0.15, 0.30, 140, 220, 15, 45, 270, 320, 20, 45, 500, 20]
            # Wheat
            param_val[:14, 5] = [0.52, 0.67, 0.20, 0.35, 30, 90, 5, 25, 125, 175, 5, 25, 500, 20]
            # Rapeseed 0.5500    0.1000   40.0000   20.0000   85.0000    5.0000    0.4500    0.1000   90.0000    5.0000
            # 140.0000   10.0000
            param_val[:, 6] = [0.73, 0.78, 0.1, 0.15, 35, 40, 18, 22, 83, 87, 3, 7, 500, 20, 0.62, 0.67, 0.07, 0.13, 88,
                               92,
                               3, 7, 137, 143, 7, 13]
            # Barley
            param_val[:14, 7] = [0.52, 0.67, 0.20, 0.35, 30, 90, 5, 25, 120, 170, 5, 25, 500, 20]
            # Bi-culture: wheat-soy
            param_val[:, 8] = [0.5, 0.55, 0.10, 0.15, 45, 55, 10, 15, 130, 135, 2, 5, 500, 20, 0.65, 0.7, 0.10, 0.15,
                               170,
                               180, 10, 15, 280, 300, 25, 35]
            # Evergreen
            param_val[:14, 9] = [0.01, 0.015, 0.55, 0.7, 0, 365, 100, 150, 0, 365, 100, 150, 500, 20]
            # Decideous
            param_val[:14, 10] = [0.2, 0.35, 0.40, 0.5, 23, 27, 15, 20, 315, 320, 15, 20, 500, 20]
            # Water
            param_val[:14, 11] = [0.01, 0.02, -0.2, 0, 150, 200, 10, 15, 50, 200, 10, 15, 500, 20]
            # Build
            param_val[:14, 12] = [0.01, 0.02, 0.2, 0.3, 0, 365, 10, 15, 0, 365, 10, 15, 500, 20]

        elif option == 3:

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
            #param_val[:14, 6] = [0.70, 0.8, 0.05, 0.20, 30, 45, 15, 25, 80, 90, 3, 12, 500, 10]
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

        return param_val, class_names
