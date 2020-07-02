#from EvalAlgo.EvalAlgo import *
from GenLabelNoiseTS.GenLabelNoiseTS import *
import numpy as np

pathTwoClass = './data/TwoClass/'
pathFiveClass = './data/FiveClass/'
pathTenClass = './data/TenClass/'

def main():
    systematicChange = False
    nbclass = 2
    seed = 0

    # EvalAlgo(pathTwoClass, nbclass, seed, systematicChange)
    # visualisationEval(2, None, False)
    # visualisationEval(5, None, False)
    # visualisationEval(5, None, True)
    # visualisationEval(10, None, False)

    classList = ('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy',
                 'Wheat', 'Rapeseed', 'Barley',
                 'Evergreen', 'Decideous')
    generator = GenLabelNoiseTS(filename="dataFrame.h5", dir='src/file/', classList=classList, csv=True,
                                verbose=True)
    generator.defaultVisualisation(dir='./results/plots/')


if __name__ == "__main__": main()
