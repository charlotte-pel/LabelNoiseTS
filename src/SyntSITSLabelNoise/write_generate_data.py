import numpy as np
import pandas as pd
import os

"""Module WriteGenerateData"""


class WriteGenerateData:

    @staticmethod
    def write_generate_data(class_names, param_val, dates, filename):
        # Test Filename
        if not os.path.exists(filename):
            print('Invalid filename: ' + filename)

        # Variance per parameter[A;B;x0;x1;x2;x3]

        # Double or simple sigmoid
        db_sigmo = 14. * np.ones((1, np.size(param_val[2])))
        # print(db_sigmo.shape)
        for i in range(0, np.size(param_val[2])):
            sig_param_val = param_val[15:, i]
            # print(sum(sig_param_val))
            if sum(sig_param_val) != -len(sig_param_val):
                db_sigmo[0, i] = 26

        print(db_sigmo)
