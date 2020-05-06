from src.SyntSITSLabelNoise.GeneratorNoise import *
from src.SyntSITSLabelNoise.InitParamValues import *
from src.SyntSITSLabelNoise.WriteGenerateData import *


def main():
    (param_val, class_names) = InitParamValues.initParamValues(3)
    dates = InitParamValues.generateDates()
    for i in range(0,1):
        WriteGenerateData.writeGenerateData(class_names, param_val, dates, "training_Python.txt")
        const_level_noise = [round(i, 2) for i in np.arange(0.05, 1.00, 0.05)]
        for i in const_level_noise:
            GeneratorNoise.generatorNoisePerClass("dataFrame.h5", i)
        #(nb_class, dates, class_names, samplesClass) = ReadGenerateData.readGenerateData("training_Python.txt")
        #(nb_class, dates, class_names, samplesClass) = ReadGenerateData.readGenerateDataH5('training_Python.h5')
        #(nb_class, dates, class_names, samplesClass, nbPixelClass, nbPolyClass) = ReadGenerateData.readGenerateDataH5DataFrame('dataFrame.h5')
        #ReadGenerateData.getSpeIdSpeClass('training_Python.h5','Barley',5)
        #ReadGenerateData.getSpeClass('training_Python.h5','Corn')
        #ReadGenerateData.readGenerateDataH5DataFrame('training_Python.h5')
        #df1 = pd.read_hdf('dataFrame.h5','data')
        #df2 = pd.read_hdf('dataFrame.h5', 'header')
        #print(df.loc[(df["label"]=="Wheat") & (df["polid"]==20) & (df["pixid"]==3)])
        #Drawprofils.drawMeanProfilOneClass(6, dates, class_names, samplesClass)
        #Drawprofils.drawProfilMeanSpeClass((6,), nb_class, dates, class_names, samplesClass,1)
        #Drawprofils.draw20RandomIdProfilOneClass(6, dates, class_names, samplesClass)
        #Drawprofils.drawProfilMeanSpeClass((5,6,7),nb_class, dates, class_names, samplesClass)
        #Drawprofils.drawProfilClass(class_names[6], nb_class, dates, class_names, samplesClass)
        #Drawprofils.drawProfilMeanSpeClass((0, 1, 2, 3, 4, 5, 6, 7, 9, 10), nb_class, dates, class_names, samplesClass)
        #Drawprofils.draw20RandomIdProfilOneClass(6, dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanSpeClass((6, 8), nb_class, dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanClass(nb_class,dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanSpeClass((0,1,2,3,4,5,6,7,9,10), nb_class, dates, class_names, samplesClass)
    # Drawprofils.drawProfilMeanSpeClass((6,),nb_class, dates, class_names, samplesClass)
    #Drawprofils.drawMeanProfilOneClass(6, dates, class_names, samplesClass)
    # Drawprofils.draw20RandomIdProfilOneClass(6, dates, class_names, samplesClass)


if __name__ == "__main__": main()
