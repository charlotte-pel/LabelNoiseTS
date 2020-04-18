import numpy as np
import os
from scipy.stats import norm

"""Module WriteGenerateData"""


class WriteGenerateData:

    @staticmethod
    def write_generate_data(class_names, param_val, dates, filename):
        # Test Filename
        if not os.path.exists(filename):
            print('Invalid filename: ' + filename)

        fid = open(filename, "w")

        # Variance per parameter[A;B;x0;x1;x2;x3]

        # Double or simple sigmoid
        db_sigmo = 14 * np.ones((1, np.size(param_val[2])))
        # print(db_sigmo.shape)
        for i in range(0, np.size(param_val[2])):
            sig_param_val = param_val[15:, i]
            # print(sum(sig_param_val))
            if sum(sig_param_val) != -len(sig_param_val):
                db_sigmo[0, i] = 26
        # print(db_sigmo)
        for add in range(0, np.size(param_val[2])):
            #print(add)
            data = param_val[0:int(db_sigmo[0, add]), add]
            # print(len(data))
            nb_Samples = data[12]
            polygonSize = data[11]

            # Generate the exact number of samples per polygons
            nb_Samples_Polygons = []
            while sum(nb_Samples_Polygons) < nb_Samples:
                val = max(round(1. / 5. * polygonSize * np.random.randn() + polygonSize), 1)
                nb_Samples_Polygons.append(val)
            # print(nb_Samples_Polygons)
            cmpt = 0
            # end = len(nb_Samples_Polygons)
            while sum(nb_Samples_Polygons) != nb_Samples:
                nb_Samples_Polygons[-1 - cmpt] = nb_Samples_Polygons[-1 - cmpt] - 1
                cmpt = cmpt + 1
                if cmpt == len(nb_Samples_Polygons):
                    cmpt = 0

            if db_sigmo[0, add] == 14:
                range_param = data[0:-1 - 1]
                # print(range_param)
                range_param = np.reshape(range_param, (6, 2))
            else:
                range_param = data[0:12] + data[15:]

                # print(range_param)
                range_param = np.reshape(range_param, (12, 2))

            for i in range(0, len(nb_Samples_Polygons)):
                diff_pos = 0
                sigmo_param = WriteGenerateData.generate_double_sigmo_parameters(range_param)
                #print((sigmo_param))
                if db_sigmo[0, add] == 14:
                    #print(sigmo_param)
                    x0 = sigmo_param[0, 3]

                    # Calcul de x2
                    if 1:
                        x2 = sigmo_param[0,4]
                        #print(x2)
                        mu_gauss = np.random.randint(50, 150) + x2
                        sigma_gauss = 16 * np.random.randn() + 32
                        A_gauss = np.random.rand() / 3
                        gauss = A_gauss * np.exp(-(dates - np.mean(dates)) / (2 * sigma_gauss * sigma_gauss))
                        print(gauss)
                        pos_mean = np.where(dates <= np.mean(dates))
                        pos_x2 = np.where(dates <= mu_gauss)
                        #print(pos_mean[0])
                        #print(pos_x2[0][-1])
                        diff_pos = pos_x2[0][-1] - pos_mean[0][-1]
                        if diff_pos < 0:
                            diff_pos = pos_mean[0][-1] - pos_x2[0][-1]

                    norm_pdf = norm.pdf(dates, x0, 10) + (0.05 - max(norm.pdf(dates, x0, 10)))
                    init_profil = WriteGenerateData.sigmoProfil(sigmo_param, dates)
                    if WriteGenerateData.strcmp(class_names[add], 'Evergreen') | WriteGenerateData.strcmp(
                            class_names[add], 'Build'):
                        norm_pdf = max(norm_pdf) * np.ones(1, len(dates))
                else:
                    x0 = sigmo_param[3]
                    x0bis = sigmo_param[9]
                    norm_pdf1 = norm.pdf(dates, x0, 10) + (0.05 - max(norm.pdf(dates, x0, 10)))
                    norm_pdf2 = norm.pdf(dates, x0bis, 10) + (0.05 - max(norm.pdf(dates, x0bis, 10)))
                    norm_pdf = (norm_pdf1 + norm_pdf2) / 2
                    init_profil = WriteGenerateData.doubleSigmoProfil(sigmo_param, dates)

                for j in range(1, nb_Samples_Polygons[i]):
                    # Generate variability : 2eme version
                    profil = norm_pdf * np.random.rand(1, len(init_profil)) + init_profil
                    if db_sigmo[add] == 14 & 1:
                        if len(dates) - diff_pos + 1 > diff_pos:
                            profil = profil + [gauss[len(dates) - diff_pos + 1:-1]+gauss[1:diff_pos]+
                                               gauss[diff_pos + 1:len(dates) - diff_pos]]
                        else:
                            profil = profil + [gauss[len(dates) - diff_pos + 1:diff_pos]+gauss[diff_pos + 1:-1]+
                                               gauss[1:len(dates) - diff_pos]]

                    profil = max(-np.ones(np.size(profil)), profil)
                    profil = min(np.ones(np.size(profil)), profil)

                    # Write samples
                    under_ind = '_' in class_names[1, add]
                    WriteGenerateData.fprintf(fid, 'c%s;id%u;', class_names[1, add][1: under_ind(1) - 1], i)
                    WriteGenerateData.fprintf(fid, '%d;', profil);
                    WriteGenerateData.fprintf(fid, '\n');
        fid.close()

    @staticmethod
    def fprintf(stream, format_spec, *args):
        stream.write(format_spec % args)

    @staticmethod
    def strcmp(str1, str2):
        if str1 == str2:
            return True
        else:
            return False

    @staticmethod
    def sigmoProfil(samples_sigmo_param, dates):
        profil = samples_sigmo_param[1] * (
                1 / (1 + np.exp((samples_sigmo_param[3] - dates) / samples_sigmo_param[4])) - 1 / (
                1 + np.exp((samples_sigmo_param[5] - dates) / samples_sigmo_param[6]))) + samples_sigmo_param[2]
        return profil

    @staticmethod
    def doubleSigmoProfil(samples_sigmo_param, dates):
        profil1 = samples_sigmo_param[0] * (
                1 / (1 + np.exp((samples_sigmo_param[2] - dates) / samples_sigmo_param[3])) - 1 / (
                1 + np.exp((samples_sigmo_param[4] - dates) / samples_sigmo_param[5]))) + samples_sigmo_param[1]
        profil2 = samples_sigmo_param[6] * (
                1 / (1 + np.exp((samples_sigmo_param[8] - dates) / samples_sigmo_param[9])) - 1 / (
                1 + np.exp((samples_sigmo_param[10] - dates) / samples_sigmo_param[11]))) + samples_sigmo_param[7]
        return profil1 + profil2

    @staticmethod
    def generate_double_sigmo_parameters(range_param):
        """
        :param range_param: contains min and max value for the six params
               of the double sigmoid [A ; B ; x0 ; x1 ; x2 ; x3]
        :return: give the six double_sigmoid parameters
                [A ; B ; x0 ; x1 ; x2 ; x3]
        """
        #print(range_param)
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
        #print(sigmo_param)
        return sigmo_param
