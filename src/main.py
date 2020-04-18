from src.SyntSITSLabelNoise.Write_generate_data import *
from src.SyntSITSLabelNoise.InitParamValues import *

def main():
    (param_val, class_names) = InitParamValues.initParamValues(1)
    dates = InitParamValues.generateDates()
    WriteGenerateData.write_generate_data(class_names, param_val, dates, "test")


if __name__ == "__main__": main()
