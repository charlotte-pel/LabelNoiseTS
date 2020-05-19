import os
import shutil
from GenLabelNoiseTS.GeneratorData import *


def main():
    initPath = 'initFile.csv'
    path = ('data/TwoClass', 'data/FiveClass', 'data/TenClass')
    if os.path.isdir('data/'):
        print('Dir data/ already exist !')
        print('Do you want delete dir data/ ? [y(Yes) or n(No)]')
        answer = input()
        while answer != 'y' and answer != 'n':
            answer = input()
        if answer == 'y':
            shutil.rmtree('data/')
        elif answer == 'n':
            print('Error Dir data/ already exist !')
    else:
        answer = 'y'

    if answer == 'y':
        os.mkdir('data/', 0o755)
        noiseArray = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
        l = 0
        lMax = (len(path) * 10 * len(noiseArray)) + (10 * len(noiseArray)) -1
        print('PROGRESS BAR GENERATE DATA :')
        for i in path:
            os.mkdir(i, 0o755)
            for j in range(10):
                os.mkdir(i + '/Run' + str(j + 1), 0o755)
                if i == 'data/TwoClass':
                    generator = GeneratorData(filename="dataFrame.h5", rep=i + '/Run' + str(j + 1) + '/',
                                              pathInitFile=initPath, classList=('Corn', 'Corn_Ensilage'),
                                              csv=True, verbose=False, )
                elif i == 'data/FiveClass':
                    generator = GeneratorData(filename="dataFrame.h5", rep=i + '/Run' + str(j + 1) + '/',
                                              pathInitFile=initPath,
                                              classList=('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy'),
                                              csv=True, verbose=False, )
                    a = {'Corn': 'Corn_Ensilage', 'Corn_Ensilage': 'Sorghum', 'Sorghum': 'Sunflower', 'Sunflower': 'Soy',
                         'Soy': 'Corn'}
                elif i == 'data/TenClass':
                    generator = GeneratorData(filename="dataFrame.h5", rep=i + '/Run' + str(j + 1) + '/',
                                              pathInitFile=initPath,
                                              classList=('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy',
                                                         'Wheat', 'Rapeseed', 'Barley',
                                                         'Evergreen', 'Decideous'),
                                              csv=True, verbose=False, )
                for k in noiseArray:
                    generator.getNoiseDataXY(k)
                    printProgressBar(l, lMax)
                    if i == 'data/FiveClass':
                        generator.getNoiseDataXY(k, a)
                        l += 1
                    l += 1
            generator.getTestData()
        print('DONE !')


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__": main()
