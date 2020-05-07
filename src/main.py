from SyntSITSLabelNoise.GenerateData import *
from SyntSITSLabelNoise.GeneratorNoise import *
from SyntSITSLabelNoise.InitParamValues import *
from SyntSITSLabelNoise.WriteGenerateData import *


def main():
    (param_val, class_names) = InitParamValues.initParamValues(3)
    dates = InitParamValues.generateDates()
    for i in range(0, 1):
        (dfHeader, dfData) = GenerateData.generateData(class_names, param_val, dates)
        WriteGenerateData.writeGenerateDataToH5("dataFrame.h5", dfHeader, dfData)
        const_level_noise = [round(i, 2) for i in np.arange(0.05, 1.00, 0.05)]
        for i in const_level_noise:
            GeneratorNoise.generatorNoisePerClass("dataFrame.h5", i)


if __name__ == "__main__": main()
