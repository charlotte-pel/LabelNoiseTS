import os

import numpy as np
from scipy.stats import norm


class WriteGenerateData:
    """
    Class WriteGenerateData
    The class contains only static method/function
    """

    @staticmethod
    def write_generate_data(class_names, param_val, dates, filename):
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

        # Variance per parameter[A;B;x0;x1;x2;x3]
        # var_param = [0.001 0.01 7 0.2 2 0.2];

        # Double or simple sigmoid
        db_sigmo = 14 * np.ones((1, np.size(param_val[2])))
        for i in range(0, np.size(param_val[2])):
            sig_param_val = param_val[15:, i]
            if sum(sig_param_val) != -len(sig_param_val):
                db_sigmo[0, i] = 26

        # Open the file
        fid = open(filename, "w")

        # Write the header
        # N = nbClass
        # dates
        # class1;nbSamples
        # ...
        # % classN;nbSamples
        WriteGenerateData.fprintf(fid, str(np.size(param_val[2]) - 1))
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

        for add in range(0, np.size(param_val[2])):
            data = param_val[0:int(db_sigmo[0, add]), add]
            nb_Samples = data[12]
            polygonSize = data[11]

            # Generate the exact number of samples per polygons
            nb_Samples_Polygons = []
            while sum(nb_Samples_Polygons) < nb_Samples:
                val = max(round(1. / 5. * polygonSize * np.random.randn() + polygonSize), 1)
                nb_Samples_Polygons.append(val)

            cmpt = 0
            while (sum(nb_Samples_Polygons) != nb_Samples) & ((len(nb_Samples_Polygons) - 1 - cmpt) >= 0):
                nb_Samples_Polygons[-1 - cmpt] = nb_Samples_Polygons[-1 - cmpt] - 1
                cmpt = cmpt + 1
                if cmpt == len(nb_Samples_Polygons):
                    cmpt = 0

            if db_sigmo[0, add] == 14:
                range_param = data[:-2]
                range_param = np.reshape(range_param, (6, 2))
            else:
                range_param = [*data[:12], *data[14:]]
                range_param = np.reshape(range_param, (12, 2))

            for i in range(0, len(nb_Samples_Polygons)):
                diff_pos = 0
                sigmo_param = WriteGenerateData.generate_double_sigmo_parameters(range_param)
                gauss = []

                if db_sigmo[0, add] == 14:
                    x0 = sigmo_param[0, 2]

                    # Calcul of x2
                    # Date of the inflection point of the main crop.Page 72 - 73 manuscript.
                    x2 = sigmo_param[0, 4]

                    # Average dates of regrowth. The regrowth curve is offset in relation to the main crop (for
                    # winter crops the regrowth is in summer and vice versa for winter crops).
                    mu_gauss = np.random.randint(50,
                                                 151) + x2

                    # Standard deviation of regrowth
                    sigma_gauss = 16 * np.random.rand() + 32

                    # Amplitude of regrowth
                    A_gauss = np.random.rand() / 3

                    # Matlab code for gauss
                    # gauss = A_gauss.*exp(-(dates-mean(dates)).*(dates-mean(dates))./(2.*sigma_gauss.*sigma_gauss));
                    gauss = A_gauss * np.exp(
                        -(dates - np.mean(dates)) * (dates - np.mean(dates)) / (2 * sigma_gauss * sigma_gauss))

                    # TODO comments
                    pos_mean = np.where(dates <= np.mean(dates))
                    pos_x2 = np.where(dates <= mu_gauss)
                    diff_pos = pos_x2[0][-1] - pos_mean[0][-1]
                    if diff_pos < 0:
                        diff_pos = pos_mean[0][-1] - pos_x2[0][-1]

                    # Matalb code for norm_pdf
                    # norm_pdf = normpdf(dates,x0,10) + (0.05-max(normpdf(dates,x0,10)))
                    norm_pdf = norm.pdf(dates, x0, 10) + (0.05 - max(norm.pdf(dates, x0, 10)))
                    init_profil = WriteGenerateData.sigmoProfil(sigmo_param, dates)
                    if WriteGenerateData.strcmp(class_names[add], 'Evergreen') | WriteGenerateData.strcmp(
                            class_names[add], 'Build'):
                        norm_pdf = max(norm_pdf) * np.ones((1, len(dates)))
                else:
                    # NormaPDF -> TODO
                    x0 = sigmo_param[0, 3]
                    x0bis = sigmo_param[0, 9]
                    norm_pdf1 = norm.pdf(dates, x0, 10) + (0.05 - max(norm.pdf(dates, x0, 10)))
                    norm_pdf2 = norm.pdf(dates, x0bis, 10) + (0.05 - max(norm.pdf(dates, x0bis, 10)))
                    norm_pdf = (norm_pdf1 + norm_pdf2) / 2
                    init_profil = WriteGenerateData.doubleSigmoProfil(sigmo_param, dates)

                for j in range(1, int(nb_Samples_Polygons[i])):
                    # Generate variability : 2eme version
                    # Code Matlab
                    # profil = norm_pdf.*randn(1,length(init_profil)) + init_profil;
                    profil = norm_pdf * np.random.randn(1, len(init_profil)) + init_profil
                    if db_sigmo[0][add] == 14:
                        if len(dates) - diff_pos > diff_pos:
                            # Origin code Matlab: profil = profil + [gauss(length(dates)-diff_pos+1:end) gauss(
                            # 1:diff_pos) gauss(diff_pos+1:length(dates)-diff_pos)] Pour ajouter des tab à la suite:
                            # gauss2 = [*l1,*l2,*l3]
                            gauss2 = np.array([*gauss[(len(dates) - diff_pos):], *gauss[:diff_pos],
                                               *gauss[diff_pos:len(dates) - diff_pos]]).reshape(1, 15)
                            profil[0] = profil[0] + gauss2
                        else:
                            # Origin code Matlab: profil = profil + [gauss(length(dates)-diff_pos+1:diff_pos) gauss(
                            # diff_pos+1:end) gauss(1:length(dates)-diff_pos)]
                            gauss2 = np.array([*gauss[(len(dates) - diff_pos):diff_pos], *gauss[diff_pos:],
                                               *gauss[:len(dates) - diff_pos]]).reshape(1, 15)
                            profil[0] = profil[0] + gauss2

                    tmpMatrix = -np.ones(np.size(profil))
                    tmpProfil = np.ndarray((1, 15))
                    for k in range(0, np.size(profil)):
                        tmpProfil[0, k] = (max(tmpMatrix[k], profil[0, k]))
                    profil = tmpProfil
                    tmpMatrix = np.ones(np.size(profil))
                    tmpProfil = np.ndarray((1, 15))
                    for k in range(0, np.size(profil)):
                        tmpProfil[0, k] = (min(tmpMatrix[k], profil[0, k]))
                    profil = tmpProfil

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
                1 + np.exp((samples_sigmo_param[0, 4] - dates) / samples_sigmo_param[0, 5]))) + samples_sigmo_param[
                     0, 1]
        return profil

    @staticmethod
    def doubleSigmoProfil(samples_sigmo_param, dates):
        """

        :param samples_sigmo_param:
        :param dates: number of days since New Year's Day
               dates = [0,25,50,...]
        :return: a double sigmo profil
        """
        profil1 = samples_sigmo_param[0, 0] * (
                1 / (1 + np.exp((samples_sigmo_param[0, 2] - dates) / samples_sigmo_param[0, 3])) - 1 / (
                1 + np.exp((samples_sigmo_param[0, 4] - dates) / samples_sigmo_param[0, 5]))) + samples_sigmo_param[
                      0, 1]
        profil2 = samples_sigmo_param[0, 6] * (
                1 / (1 + np.exp((samples_sigmo_param[0, 8] - dates) / samples_sigmo_param[0, 9])) - 1 / (
                1 + np.exp((samples_sigmo_param[0, 10] - dates) / samples_sigmo_param[0, 11]))) + samples_sigmo_param[
                      0, 7]
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
            mu = (range_param[i, 0] + range_param[i, 1] / 2.0)
            sqrt_var = (range_param[i, 1] - mu) / 3.0
            val = sqrt_var * np.random.randn() + mu
            while val < range_param[i, 0] or val > range_param[i, 1]:
                val = sqrt_var * np.random.randn() + mu
            sigmo_param[0, i] = val
        return sigmo_param
