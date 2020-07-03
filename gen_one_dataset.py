import argparse
import ast

from GenLabelNoiseTS.GenLabelNoiseTS import *
import sys

# python gen_one_dataset.py -d src/file/ -f dataset.h5 -noClass 10 -noise random -noiseLevel [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis src/img/
# If you use dict don't put any space!!!
# python gen_one_dataset.py -d src/file/ -f dataset.h5 -noClass 10 -noise {'Wheat':('Barley','Soy'),'Barley':'Soy'} -noiseLevel [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis src/img

parser = argparse.ArgumentParser()
# Required arguments:
parser.add_argument('-d', required=True, type=str, dest='dir', action='append', help="directory")
parser.add_argument('-f', required=True, type=str, dest='fileName', action='append', help="File name")
parser.add_argument('-noClass', required=True, type=int, dest='noClass', action='append', help="Number of class")
parser.add_argument('-noise', required=True, dest='noise', action='append', help="Noise type")
parser.add_argument('-noiseLevel', required=True, dest='noiseLevel', action='append', help="Noise level")

# Optional arguments:
parser.add_argument('-save_csv', required=False, dest='save_csv', action='store_true', help="Option csv")
parser.add_argument('-v', required=False, dest='verbose', action='store_true', help="Verbose Mode")
parser.add_argument('-vis', required=False, type=str, dest='visualisation', action='append', help="Default visualisation")

args = parser.parse_args()
dir = args.dir[0]
fileName = args.fileName[0]
noClass = args.noClass[0]
noise = args.noise[0]
noiseLevel = args.noiseLevel[0]

if args.save_csv:
    save_csv = True
else:
    save_csv = False
if args.verbose:
    verbose = True
else:
    verbose = False
if args.visualisation:
    visualisation = args.visualisation[0]
else:
    visualisation = False

if noClass == 2:
    classList = ('Corn', 'Corn_Ensilage')
elif noClass == 5:
    classList = ('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy')
elif noClass == 10:
    classList = ('Corn', 'Corn_Ensilage', 'Sorghum', 'Sunflower', 'Soy',
                 'Wheat', 'Rapeseed', 'Barley',
                 'Evergreen', 'Decideous')
else:
    print('Error no class !!!')
    print('Only 2,5 or 10 class')
    sys.exit(0)

generator = GenLabelNoiseTS(filename=fileName, classList=classList, csv=save_csv, verbose=verbose, dir=dir)

if visualisation is not False:
    generator.defaultVisualisation(dir=visualisation)

if noise == 'random':
    noise = None
else:
    noise = ast.literal_eval(noise)

noiseLevel = ast.literal_eval(noiseLevel)

for i in noiseLevel:
    generator.getNoiseDataXY(i, noise)
generator.getTestData()
