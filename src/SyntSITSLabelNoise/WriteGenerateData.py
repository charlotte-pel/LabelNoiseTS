import os

import numpy as np


class WriteGenerateData:
    """
    Class WriteGenerateData
    The class contains only static method/function
    """

    @staticmethod
    def writeGenerateData(class_names, param_val, dates, filename):
        """
        :param class_names: contains the names of different class
               class_names = ['Corn', 'Corn_ensilage',...]
        :param param_val: param of double sigmo for each class
               param_val[:14, 0] = [0.62, 0.67, 0.2, 0.25, 122, 182, 5, 20, 270, 290, 15, 20, 500, 20]
        :param dates: number of days since New Year's Day
               dates = [0,25,50,...]
        :param filename: the name of the file in which the data will be entered
        :return: No return the ouput is the file.
        """
        # Test Filename
        if not os.path.exists(filename):
            print('Invalid filename: ' + filename)

        # We transpose the list of param_val to be able to write it correctly (on line) in the file.
        param_val_transpose = np.transpose(param_val)

        # Double or simple sigmoid
        # Reshape is to put array like this [[14,...,26,...,14,...]] instead of [14,...,26,...,14,...]
        db_sigmo = np.where(np.sum(param_val[14:, ], axis=0) != -len(param_val[14:, ]), 26, 14).reshape(
            (1, np.size(param_val[2])))

        # Open the file
        fid = open(filename, "w")

        # Write the header
        # N = nbClass
        # dates
        # class1;nbSamples
        # ...
        # % classN;nbSamples
        WriteGenerateData.fprintf(fid, str(np.size(param_val[2])))
        WriteGenerateData.fprintf(fid, '\n')
        for i in dates:
            WriteGenerateData.fprintf(fid, '%d;', i)
        WriteGenerateData.fprintf(fid, '\n')
        for i in range(0, np.size(param_val[2]) - 1):
            WriteGenerateData.fprintf(fid, 'c%s;%s;', i + 1, class_names[i])
            for j in param_val_transpose[i]:
                if j != -1:
                    WriteGenerateData.fprintf(fid, '%0.2f;', j)
            WriteGenerateData.fprintf(fid, '\n')

        # Generate Data Start:
        for add in range(0, np.size(param_val[2])):
            print(add)
            data = param_val[0:int(db_sigmo[0, add]), add]
            nb_Samples = data[12]
            polygonSize = data[13]

            # Generate the exact number of samples per polygons
            # (same number of samples per polygon)
            nb_Samples_Polygons = polygonSize * np.ones((1, int(np.floor(nb_Samples / polygonSize))))

            if db_sigmo[0, add] == 14:
                range_param = data[:-2]
                range_param = np.reshape(range_param, (6, 2))
            else:
                range_param = [*data[:12], *data[14:]]
                range_param = np.reshape(range_param, (12, 2))

            for i in range(0, len(nb_Samples_Polygons[0])):  # id poly

                sigmo_param = WriteGenerateData.generate_double_sigmo_parameters(range_param)
                # Calculation of Gaussian parameters

                # Calculation of x2
                # Date of the inflection point of the main crop.Page 72 - 73 manuscript.
                if db_sigmo[0, add] == 14:
                    x2 = sigmo_param[0, 4]
                else:
                    x2 = sigmo_param[0, 10]

                # Average dates of regrowth. The regrowth curve is offset in relation to the main crop (for
                # winter crops the regrowth is in summer and vice versa for winter crops).
                mu_gauss = np.random.randint(50, 151) + x2

                # Standard deviation of regrowth
                sigma_gauss = 16 * np.random.rand() + 32

                # Amplitude of regrowth
                A_gauss = np.random.rand() / 3

                if WriteGenerateData.strcmp(class_names[add], 'Evergreen') | WriteGenerateData.strcmp(
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

                for j in range(1, int(nb_Samples_Polygons[0, i])):  # pixel poly
                    # Generate variability : 3eme version

                    reduce_sigmo_param = np.zeros((len(sigmo_param[0]), 2))
                    variability = np.random.randint(5, 21)
                    reduce_sigmo_param[:, 0] = sigmo_param - (range_param[:, 1] - range_param[:, 0]) / variability
                    reduce_sigmo_param[:, 1] = sigmo_param + (range_param[:, 1] - range_param[:, 0]) / variability

                    reduce_sigmo_param[:, 0] = np.where(reduce_sigmo_param[:, 0] < range_param[:, 0], range_param[:, 0],
                                                        reduce_sigmo_param[:, 0])
                    reduce_sigmo_param[:, 1] = np.where(reduce_sigmo_param[:, 1] > range_param[:, 1], range_param[:, 1],
                                                        reduce_sigmo_param[:, 1])

                    sample_sigmo_param = WriteGenerateData.generate_double_sigmo_parameters(reduce_sigmo_param)

                    if db_sigmo[0, add] == 14:
                        init_profil = WriteGenerateData.sigmoProfil(sample_sigmo_param, dates)
                    else:
                        init_profil = WriteGenerateData.doubleSigmoProfil(sample_sigmo_param, dates)

                    norm_pdf = 0.05 * (2 * np.random.rand(1, len(dates)))
                    vec_noisy_dates = np.random.permutation(np.random.randint(0, len(dates), size=len(dates)))
                    norm_pdf[0, vec_noisy_dates[:np.random.randint(0, len(dates))]] = 0
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

                    # Write samples
                    WriteGenerateData.fprintf(fid, 'c%s;id%u;', add + 1, i + 1)
                    for k in profil[0]:
                        WriteGenerateData.fprintf(fid, '%f;', k)
                    WriteGenerateData.fprintf(fid, '\n')
        fid.close()

    @staticmethod
    def fprintf(stream, format_spec, *args):
        """
        Function to format the string and write in the file
        :param stream: the file
        :param format_spec: format of the string which write
               Ex: 'c%s;id%u;'
        :param args: string to add in the file
        :return: No return -> add the line to the file
        """
        stream.write(format_spec % args)

    @staticmethod
    def strcmp(str1, str2):
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
    def sigmoProfil(samples_sigmo_param, dates):
        """
        TODO Correction code: utiliser sigmoProfil dans doubleSigmoProfil
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
    def doubleSigmoProfil(samples_sigmo_param, dates):
        """

        :param samples_sigmo_param: [A ; B ; x0 ; x1 ; x2 ; x3]
        :param dates: number of days since New Year's Day
               dates = [0,25,50,...]
        :return: a double sigmo profil
        """
        profil1 = WriteGenerateData.sigmoProfil(samples_sigmo_param[:, :6], dates)
        profil2 = WriteGenerateData.sigmoProfil(samples_sigmo_param[:, 6:], dates) + 0.05
        return profil1 + profil2

    @staticmethod
    def generate_double_sigmo_parameters(range_param):
        """

        :param range_param: contains min and max value for the six params
               of the double sigmoid [A ; B ; x0 ; x1 ; x2 ; x3]
        :return: give the six double_sigmoid parameters
                [A ; B ; x0 ; x1 ; x2 ; x3]
        """
        sigmo_param = np.zeros((1, len(range_param)))
        for i in range(0, len(range_param)):
            if range_param[i, 0] > range_param[i, 1]:
                print('Error MIN > MAX')
            mu = (range_param[i, 0] + range_param[i, 1]) / 2.0
            sqrt_var = (range_param[i, 1] - mu) / 3.0
            val = sqrt_var * np.random.randn() + mu
            while val < range_param[i, 0] or val > range_param[i, 1]:
                val = sqrt_var * np.random.randn() + mu
            sigmo_param[0, i] = val
        return sigmo_param
