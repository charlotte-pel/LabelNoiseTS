from src.SyntSITSLabelNoise.DrawProfils import *
from src.SyntSITSLabelNoise.InitParamValues import *
from src.SyntSITSLabelNoise.ReadGenerateData import *
from src.SyntSITSLabelNoise.Write_generate_data import *


def main():
    (param_val, class_names) = InitParamValues.initParamValues(3)
    dates = InitParamValues.generateDates()
    # print(class_names)
    # print(np.shape(param_val))
    for i in range(0,10):
        WriteGenerateData.write_generate_data(class_names, param_val, dates, "training_Python.txt")
        (nb_class, dates, class_names, samplesClass) = ReadGenerateData.read_generate_data("training_Python.txt")
        #Drawprofils.drawMeanProfilOneClass(6, dates, class_names, samplesClass)
        Drawprofils.drawProfilMeanSpeClass((6,), nb_class, dates, class_names, samplesClass)
        #Drawprofils.draw20RandomIdProfilOneClass(6, dates, class_names, samplesClass)
        #Drawprofils.drawProfilMeanSpeClass((5,6,7),nb_class, dates, class_names, samplesClass)
        #Drawprofils.drawProfilClass(class_names[6], nb_class, dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanSpeClass((6, 8), nb_class, dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanClass(nb_class,dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanSpeClass((0,1,2,3,4,5,6,7,9,10), nb_class, dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanSpeClass((6,),nb_class, dates, class_names, samplesClass)
    #Drawprofils.drawMeanProfilOneClass(6, dates, class_names, samplesClass)
    # Drawprofils.draw20RandomIdProfilOneClass(6, dates, class_names, samplesClass)


if __name__ == "__main__": main()
