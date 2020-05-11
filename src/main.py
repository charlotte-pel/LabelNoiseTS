import h5py
from SyntSITSLabelNoise.GenerateData import *
from SyntSITSLabelNoise.GeneratorNoise import *
from SyntSITSLabelNoise.InitParamValues import *
from SyntSITSLabelNoise.WriteGenerateData import *
from SyntSITSLabelNoise.GeneratorData import *
import numpy as np
import os


def main():
    a = {'Wheat': ('Barley','Soy','Build'),'Barley':'Soy'}
    generator = GeneratorData("dataFrame.h5")
    (X,Y) = generator.getDataXY()
    (X,Y) = generator.getNoiseDataXY(0.05,a)
    (X,Y) = (generator.getTestData())
    os.remove("dataFrame.h5")
    #WriteGenerateData.writeHeaderNoise('dataFrame.h5')
    #ReadGenerateData.getAlreadyGenNoise('dataFrame.h5',0.05)
    # (param_val, class_names) = InitParamValues.initParamValues(3)
    # dates = InitParamValues.generateDates()
    # for i in range(0, 1):
    #     (dfHeader, dfData) = GenerateData.generateData(class_names, param_val, dates)
    #     WriteGenerateData.writeGenerateDataToH5("dataFrame.h5", dfHeader, dfData)
    #     const_level_noise = [round(i, 2) for i in np.arange(0.05, 1.00, 0.05)]
    #     for i in const_level_noise:
    #         a = {}
    #         a['Wheat'] = 'Barley'
    #         a['Barley'] = 'Soy'
    #         (noiseLevel,dfNoise,systematicChange) = GeneratorNoise.generatorNoisePerClass("dataFrame.h5", i,a)
    #         WriteGenerateData.writeGenerateNoisyData('dataFrame.h5',noiseLevel,dfNoise, systematicChange)
    #     # Show all dataFrame in h5 file:
    #     with h5py.File("dataFrame.h5", "r") as f:
    #         # List all groups
    #         print("Keys: %s" % f.keys())


if __name__ == "__main__": main()
