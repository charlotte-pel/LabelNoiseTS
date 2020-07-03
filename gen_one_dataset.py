import ast
import sys

from GenLabelNoiseTS.GenLabelNoiseTS import *


# python gen_one_dataset.py -d src/file/ -f dataset.h5 -noClass 10 -noise random -noise.level [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis
# If you use dict don't put any space!!!
# python gen_one_dataset.py -d src/file/ -f dataset.h5 -noClass 10 -noise {'Wheat':('Barley','Soy'),'Barley':'Soy'} -noise.level [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis

def main():
    initPath = 'init_param_file.csv'
    args = list(sys.argv)
    a = {}
    i = 1
    while i < len(sys.argv):
        if i < 10:
            a[str(args[i])] = args[i + 1]
            i += 2
        else:
            a[str(args[i])] = True
            i += 1
    if '-save_csv' not in list(a.keys()):
        a['-save_csv'] = False
    if '-v' not in list(a.keys()):
        a['-v'] = False
    if '-vis' not in list(a.keys()):
        a['-vis'] = False

    if a['-noClass'] == str(2):
        classList = ('Corn', 'Corn_Ensilage')
    elif a['-noClass'] == str(5):
        classList = ('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy')
    elif a['-noClass'] == str(10):
        classList = ('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy',
                     'Wheat', 'Rapeseed', 'Barley',
                     'Evergreen', 'Decideous')
    else:
        print('Error no class !!!')
        print('Only 2,5 or 10 class')
        sys.exit(0)

    generator = GenLabelNoiseTS(dir=a['-d'], filename=a['-f'], classList=classList, csv=a['-save_csv'], verbose=a['-v'])
    (X, Y) = generator.getDataXY()
    if a['-vis'] is True:
        generator.defaultVisualisation(dir='./results/plots/')
    a['-noise.level'] = a['-noise.level'].replace('[', '')
    a['-noise.level'] = a['-noise.level'].replace(']', '')
    a['-noise.level'] = list(a['-noise.level'].split(","))
    a['-noise.level'] = [float(i) for i in a['-noise.level']]
    if a['-noise'] == 'random':
        a['-noise'] = None
    else:
        a['-noise'] = ast.literal_eval(a['-noise'])
    for i in a['-noise.level']:
        (X, Y) = generator.getNoiseDataXY(i, a['-noise'])
    (X, Y) = generator.getTestData()


if __name__ == "__main__": main()
