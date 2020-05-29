import random
import numpy as np
import pandas as pd


class GenerateData:
    """
    Class GenerateData
    The class contains only static method/function
    """

    @staticmethod
    def generateData(seed, initFilename, classList=None):
        """
        :param seed:
        :param initFilename:
        :return: No return the ouput is the file.
        """

        # class_names: contains the names of different class
        #                class_names = ['Corn', 'Corn_ensilage',...]
        # param_val: param of double sigmo for each class
        #                param_val[:14, 0] = [0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
        # dates: number of days since New Year's Day
        #                dates = [0,25,50,...]

        # Set random_state:
        randomState = np.random.RandomState(seed)
        # Get class names, param, dates from initFile.
        param_val = pd.DataFrame(pd.read_csv(initFilename))

        if classList is None:
            class_names = np.array(param_val['class_names'])
            nbClass = len(class_names) - 1
        else:
            class_names = classList
            nbClass = len(class_names)

        tmpClass = np.array(param_val['class_names'])[:-1]
        for i in tmpClass:
            if i not in class_names:
                param_val = param_val.drop(param_val[param_val['class_names'] == i].index)
        param_val = param_val.reset_index()
        del param_val['index']

        del param_val['class_names']
        dates = np.array(param_val.loc[param_val.index[-1], :])
        dates = dates[np.logical_not(np.isnan(dates))]
        param_val = param_val.drop(param_val.index[-1])
        param_val = param_val.fillna(-1)
        param_val = np.array(param_val)
        # Double or simple sigmoid
        db_sigmo = np.where(np.sum(param_val[:, 14:], axis=1) != -param_val[:,14:].shape[1], 26, 14)

        unique_sequencePolid = GenerateData._uniqueid(randomState)
        unique_sequencePixid = GenerateData._uniqueid(randomState)

        dfHeader = []
        tmpDataFrame = []
        tmpProfils = []
        # Write the header
        # N = nbClass
        # dates
        # class1;nbSamples
        # ...
        # % classN;nbSamples
        dfHeader.append(np.array(dates))
        dfHeader.append(np.array({'dataSeed': seed}))
        for i in range(0, nbClass):
            dfHeader.append(np.array([class_names[i], param_val[i]]))
        dfHeader = pd.DataFrame(np.array(dfHeader))

        # Generate Data Start:
        for add in range(0, nbClass):
            data = param_val[add, :int(db_sigmo[add])]
            nb_Samples = data[0]
            polygonSize = data[1]

            # Generate the exact number of samples per polygons
            # (same number of samples per polygon)
            nb_Samples_Polygons = polygonSize * np.ones((1, int(np.floor(nb_Samples / polygonSize))))

            range_param = data[2:]
            if db_sigmo[add] == 14:
                range_param = np.reshape(range_param, (6, 2))
            else:
                range_param = np.reshape(range_param, (12, 2))

            for i in range(0, len(nb_Samples_Polygons[0])):  # id poly
                polid = next(unique_sequencePolid)
                sigmo_param = GenerateData._generate_double_sigmo_parameters(range_param, randomState)
                # Calculation of Gaussian parameters

                # Calculation of x2
                # Date of the inflection point of the main crop.Page 72 - 73 manuscript.
                if db_sigmo[add] == 14:
                    x2 = sigmo_param[0, 4]
                else:
                    x2 = sigmo_param[0, 10]

                # Average dates of regrowth. The regrowth curve is offset in relation to the main crop (for
                # winter crops the regrowth is in summer and vice versa for winter crops).
                mu_gauss = randomState.randint(50, 151) + x2

                # Standard deviation of regrowth
                sigma_gauss = 16 * randomState.rand() + 32

                # Amplitude of regrowth
                A_gauss = randomState.rand() / 3

                if GenerateData._strcmp(class_names[add], 'Evergreen') | GenerateData._strcmp(
                        class_names[add], 'Decideous'):
                    A_gauss = 0

                # Matlab code for gauss
                # gauss = A_gauss.*exp(-(dates-mean(dates)).*(dates-mean(dates))./(2.*sigma_gauss.*sigma_gauss));
                gauss = A_gauss * np.exp(
                    -(dates - np.mean(dates)) * (dates - np.mean(dates)) / (2 * sigma_gauss * sigma_gauss))

                pos_mean = np.where(dates <= np.mean(dates))
                pos_x2 = np.where(dates <= mu_gauss)
                diff_pos = pos_x2[0][-1] - pos_mean[0][-1]
                if diff_pos < 0:
                    diff_pos = pos_mean[0][-1] - pos_x2[0][-1]

                for j in range(0, int(nb_Samples_Polygons[0, i])):  # pixel poly
                    # Generate variability : 3eme version

                    reduce_sigmo_param = np.zeros((len(sigmo_param[0]), 2))
                    variability = randomState.randint(5, 21)
                    reduce_sigmo_param[:, 0] = sigmo_param - (range_param[:, 1] - range_param[:, 0]) / variability
                    reduce_sigmo_param[:, 1] = sigmo_param + (range_param[:, 1] - range_param[:, 0]) / variability

                    reduce_sigmo_param[:, 0] = np.where(reduce_sigmo_param[:, 0] < range_param[:, 0], range_param[:, 0],
                                                        reduce_sigmo_param[:, 0])
                    reduce_sigmo_param[:, 1] = np.where(reduce_sigmo_param[:, 1] > range_param[:, 1], range_param[:, 1],
                                                        reduce_sigmo_param[:, 1])

                    sample_sigmo_param = GenerateData._generate_double_sigmo_parameters(reduce_sigmo_param, randomState)

                    if db_sigmo[add] == 14:
                        init_profil = GenerateData._sigmoProfil(sample_sigmo_param, dates)
                    else:
                        init_profil = GenerateData._doubleSigmoProfil(sample_sigmo_param, dates)

                    norm_pdf = 0.05 * (2 * randomState.rand(1, len(dates)))
                    vec_noisy_dates = randomState.permutation(randomState.randint(0, len(dates), size=len(dates)))
                    norm_pdf[0, vec_noisy_dates[:randomState.randint(0, len(dates))]] = 0
                    profil = norm_pdf + init_profil

                    if len(dates) - diff_pos > diff_pos:
                        # Pour ajouter des tab à la suite: gauss2 = [*l1,*l2,*l3]
                        gauss2 = np.array([*gauss[(len(dates) - diff_pos):], *gauss[:diff_pos],
                                           *gauss[diff_pos:len(dates) - diff_pos]]).reshape(1, 15)
                        profil[0] = profil[0] + gauss2
                    else:
                        # Origin code Matlab: profil = profil + [gauss(length(dates)-diff_pos+1:diff_pos) gauss(
                        # diff_pos+1:end) gauss(1:length(dates)-diff_pos)]
                        gauss2 = np.array([*gauss[(len(dates) - diff_pos):diff_pos], *gauss[diff_pos:],
                                           *gauss[:len(dates) - diff_pos]]).reshape(1, 15)
                        profil[0] = profil[0] + gauss2

                    # Return max between -np.ones((1, np.size(profil))) and profil
                    profil = np.where(-np.ones((1, np.size(profil))) > profil, -np.ones((1, np.size(profil))), profil)
                    # Return min between np.ones((1, np.size(profil))) and profil
                    profil = np.where(np.ones((1, np.size(profil))) < profil, np.ones((1, np.size(profil))), profil)
                    tmpDataFrame.append([class_names[add], polid, next(unique_sequencePixid)])
                    tmpProfils.append(profil[0])

        tmpDataFrame = np.array(tmpDataFrame)
        tmpProfils = np.array(tmpProfils)
        tmplen = len(tmpProfils[0])
        tmpProfils = np.transpose(tmpProfils)
        dfData = pd.DataFrame(tmpDataFrame, columns=['label', 'polid', 'pixid', ])
        for i in range(tmplen):
            dfData['d' + str(i + 1)] = tmpProfils[i]
        dfData = dfData.sort_values(by=['pixid'])
        return dfHeader, dfData

    #  -----------------------------------------------------------------------------------------------------------------
    #  Intern (Private) Functions of this class.
    #  For normal use, don't use them.
    #  -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _strcmp(str1, str2):
        """
        Function to compare two string with each other
        :param str1: first string
        :param str2: second string
        :return: true if and only if the both string are the same
        """
        if str1 == str2:
            return True
        else:
            return False

    @staticmethod
    def _sigmoProfil(samples_sigmo_param, dates):
        """

        :param samples_sigmo_param:
        :param dates: number of days since New Year's Day
               dates = [0,25,50,...]
        :return: a sigmo profil
        """
        # Matlab code
        # profil =  samples_sigmo_param(1) .* ...
        #         ( 1./ ( 1 + exp((samples_sigmo_param(3)-dates)./samples_sigmo_param(4)) )...
        #         - 1./ ( 1 + exp((samples_sigmo_param(5)-dates)./samples_sigmo_param(6)) ) )...
        #         + samples_sigmo_param(2);
        profil = samples_sigmo_param[0, 0] * (
                1 / (1 + np.exp((samples_sigmo_param[0, 2] - dates) / samples_sigmo_param[0, 3])) - 1 / (
                1 + np.exp((samples_sigmo_param[0, 4] - dates) / samples_sigmo_param[0, 5]))) + (samples_sigmo_param[
                                                                                                     0, 1] - 0.05)
        return profil

    @staticmethod
    def _doubleSigmoProfil(samples_sigmo_param, dates):
        """

        :param samples_sigmo_param: [A ; B ; x0 ; x1 ; x2 ; x3]
        :param dates: number of days since New Year's Day
               dates = [0,25,50,...]
        :return: a double sigmo profil
        """
        profil1 = GenerateData._sigmoProfil(samples_sigmo_param[:, :6], dates)
        profil2 = GenerateData._sigmoProfil(samples_sigmo_param[:, 6:], dates) + 0.05
        return profil1 + profil2

    @staticmethod
    def _generate_double_sigmo_parameters(range_param, randomState):
        """

        :param range_param: contains min and max value for the six params
               of the double sigmoid [A ; B ; x0 ; x1 ; x2 ; x3]
        :param randomState randomState used in this generation
        :return: give the six double_sigmoid parameters
                [A ; B ; x0 ; x1 ; x2 ; x3]
        """
        sigmo_param = np.zeros((1, len(range_param)))
        for i in range(0, len(range_param)):
            if range_param[i, 0] > range_param[i, 1]:
                print('Error MIN > MAX')
            mu = (range_param[i, 0] + range_param[i, 1]) / 2.0
            sqrt_var = (range_param[i, 1] - mu) / 3.0
            val = sqrt_var * randomState.randn() + mu
            while val < range_param[i, 0] or val > range_param[i, 1]:
                val = sqrt_var * randomState.randn() + mu
            sigmo_param[0, i] = val
        return sigmo_param

    @staticmethod
    def _uniqueid(randomState):
        """

        :return: Generate an generator for unique ID
        """
        seed = randomState.randint(10000,100000)
        while True:
            yield seed
            seed += 1
