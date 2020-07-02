# Generating synthetic time series dataset to study the influence of class label noise on classification performance

## Table of contents
* [Introduction](#Introduction)
* [Context](#Context)
* [Prerequisites](#Prerequisites)
* [Installation](#Installation)
* [Quick use](#Quick-use)
* [Generating data](#Generating-data)
* [Config File](#Config-File)
* [Evaluation of performances](#Evaluation-of-performances)
* [Visualisation](#Visualisation)
* [Results](#Results)
* [Contributing](#Contributing)
* [Contributors](#Contributors)
* [License](#License)

## Introduction

This module generates synthetic univarite time series datasets that contain different level of label noise. The generated datasets include remote sensing dataset specificity with a polygon concept. The idea is to take into account the field campaign protocols where sample labels are assigned by polygons describing for example crop fields.
This code supports the following journal and conference papers:
```
@article{pelletier2017effect,
  title={Effect of training class label noise on classification performances for land cover mapping with satellite image time series},
  author={Pelletier, Charlotte and Valero, Silvia and Inglada, Jordi and Champion, Nicolas and Marais Sicre, Claire and Dedieu, G{\'e}rard},
  journal={Remote Sensing},
  volume={9},
  number={2},
  pages={173},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}
@inproceedings{pelletier2017new,
  title={New iterative learning strategy to improve classification systems by using outlier detection techniques},
  author={Pelletier, Charlotte and Valero, Silvia and Inglada, Jordi and Dedieu, G{\'e}rard and Champion, Nicolas},
  booktitle={2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  pages={3676--3679},
  year={2017},
  organization={IEEE}
}
```

## Context

The automatic production of land cover maps obtained by the supervised classification of satellite image time series relies on the availability of accurate reference databases. In remote sensing, these reference databases come generally from several sources including field campaigns, thematic maps or photointerpreation of high spatial resolution remote sensing images. Although most of classification algorithms made the assumption that reference databases are gold standard, it is well known that they contain errors, artifacts and imprecisions. These errors lead to the presence of class label noise on both training and testing samples. In other words some samples are assigned a wrong class label.

This code generates univariate time series representing vegetation profiles (closed from Normalized Difference Vegetation Index) for various vegetation classes (mainly crops). The generation of urban and water time series profiles is also possible.
The generated datasets might be used for testing the robustness of various classification systems in a control environment where class label noise is completely known. 


## Prerequisites

The code relies on Pyton 3.8\
And use:
- Numpy
- Pandas
- Tables
- Jupyter Notebook (Only for use Jupyter Notebook)
- Matplotlib (Only for visualisation)
- Keras & Tensorflow (Only for TempCNN)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas.

```bash
pip install numpy
pip install pandas
pip install tables
pip install h5py
pip install jupyter # (Only for use Jupyter Notebook)
pip install matplotlib # (Only for visualisation)
pip install keras==2.4.3 tensorflow==2.2.0 # (Only for TempCNN)
```

## Quick use
#### For Generating data

Open and run the Jupyter notebook: `ExampleUseGeneratingData.ipynb`\
Located at the root of the project folder

#### For Evaluation

Open and run the Jupyter notebook: `ExampleUseEvaluation.ipynb`\
Located at the root of the project folder

## Generating data
```python
from GenLabelNoiseTS.GenLabelNoiseTS import *

# Example with a list of two class and systematic change.
generator = GenLabelNoiseTS(filename="dataFrame.h5", classList=('Wheat','Barley'), csv=True, verbose=True, dir="file/")
a = {'Wheat': 'Barley', 'Barley': 'Wheat'}
(X,Y) = generator.getDataXY()
(Xnoise,YNoise) = generator.getNoiseDataXY(0.05,a)
(Xtest,Ytest) = generator.getTestData()
generator.visualisation('img/')
```
#### Generating a dataset use python command below
```bash
python gen_data.py -d src/file/ -f dataFrame.h5 -nbClass 10 -noise random -noise.level [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis
# If you use dict don't put any space!!! 
# Do it like this: {'Wheat':('Barley','Soy'),'Barley':'Soy'}
python gen_data.py -d src/file/ -f dataFrame.h5 -nbClass 10 -noise {'Wheat':('Barley','Soy'),'Barley':'Soy'} -noise.level [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis
```
This command allow generating data in specific rep. Choose name file, number of classes, noise type, noise level, type of save file, verbose mode, making visualisation.\
For this command, these options are mandatory:
- -d + 'repPath' -> To specify repertory path
- -f + 'nameFile' -> To specify name file must be .h5
- -nbClass + 'nbClass' -> To specify number of classes
- -noise + random or dict -> Two noise type: random and systematic change, for systematic need using Python dictionary like {'Wheat':('Barley','Soy'),'Barley':'Soy'}
- -noise.level + [float,...] -> Array containing noise levels

These options are optional:
- -save_csv -> If you want saving data and noise data in csv file
- -v -> Verbose Mode
- -vis -> Saving default visualisation
#### Generating dataset for performance evaluation of machine learning algorithms use python command below
```bash
python Generating_a_dataset.py
```
This command allow creating a dataset like the one you can find above.
- Ten runs for each classification problem (TwoClass, FiveClass, TenClass)
- For each run 21 noise levels, begin to 0% to 100% with a step of 5%.
- For TwoClass noise type is random.
- For FiveClass noise types are random (21 noise levels) and systematic change (21 noise levels).
- For TenClass noise type is random. 
#### An example of a dataset named data is at the root of the directory
Dataset tree:
- data
  - TwoClass
    - Run1
      - data.csv
      - dataFrame.h5
      - random_0.csv
      - ...
      - random_100.csv
    - Run2
    - ...
    - Run10
  - FiveClass
    - Run1
      - data.csv
      - dataFrame.h5
      - random_0.csv
      - ...
      - random_100.csv
      - systematic_0_98494941304801395478184421979593253002.csv
      - ...
      - systematic_100_98494941304801395478184421979593253002.csv
    - Run2
    - ...
    - Run10
  - TenClass
    - Run1
      - data.csv
      - dataFrame.h5
      - random_0.csv
      - ...
      - random_100.csv
    - Run2
    - ...
    - Run10
    
## Config File

DataFrame from initFile.csv must be like below :
(Example initFile.csv has good format)
- First column must be named: class_names
- Columns names 0 and 1 contain number of samples and number of polygon for each class.
- Last row must contain dates (In the example below line 13), dates start column after class_names column.
```
      class_names    0   1      2       3  ...    21     22     23    24    25
0            Corn  500  10   0.57   0.720  ...   NaN    NaN    NaN   NaN   NaN
1   Corn_Ensilage  500  10   0.57   0.720  ...   NaN    NaN    NaN   NaN   NaN
2         Sorghum  500  10   0.62   0.770  ...   NaN    NaN    NaN   NaN   NaN
3       Sunflower  500  10   0.67   0.820  ...   NaN    NaN    NaN   NaN   NaN
4             Soy  500  10   0.67   0.820  ...   NaN    NaN    NaN   NaN   NaN
5           Wheat  500  10   0.52   0.670  ...   NaN    NaN    NaN   NaN   NaN
6        Rapeseed  500  10   0.70   0.800  ...  12.0  135.0  145.0   5.0  15.0
7          Barley  500  10   0.52   0.670  ...   NaN    NaN    NaN   NaN   NaN
8       Wheat_Soy  500  10   0.50   0.550  ...  15.0  280.0  300.0  25.0  35.0
9       Evergreen  500  10   0.01   0.015  ...   NaN    NaN    NaN   NaN   NaN
10      Decideous  500  10   0.20   0.350  ...   NaN    NaN    NaN   NaN   NaN
11          Water  500  10   0.01   0.020  ...   NaN    NaN    NaN   NaN   NaN
12          Build  500  10   0.01   0.020  ...   NaN    NaN    NaN   NaN   NaN
13            NaN    1  26  51.00  76.000  ...   NaN    NaN    NaN   NaN   NaN
```

## Evaluation of performances

Save Accuracy Scores in two csv file:
- All Accuracy scores for each run, Ex: `AccuracyCsvRF.csv`
- Average accuracy scores for all runs, Ex: `AccuracyRF.csv` (Use for evaluation visualisation)

```python
from EvalAlgo.EvalAlgo import *

pathTwoClass = './data/TwoClass/'
pathFiveClass = './data/FiveClass/'
pathTenClass = './data/TenClass/'

systematicChange = False
nbclass = 2
seed = 0

EvalAlgo(pathTwoClass, nbclass, seed, systematicChange)
visualisationEval(pathTwoClass, 'Two classes', systematicChange)
```
## Visualisation

#### Data visualisation
If 'yourPath/' is None, visualisation will not be save in external file.\
If you specify 'yourPath/', visualisation will be save in external file.

Except for visualisationProfilsMeanAllClass, you need specify a class Name besides 'yourPath/'

By default 'yourPath/' is None

```python
from GenLabelNoiseTS.GenLabelNoiseTS import *

generator = GenLabelNoiseTS(filename="dataFrame.h5", dir='pathToData' + 'Run' + str(1) + '/', csv=True,
                                verbose=False)
generator.visuTest(typePlot='mean')
generator.visuTest(typePlot='mean', className='Corn')
generator.visuTest(typePlot='all', className='Corn')
generator.visuTest(typePlot='random', className='Corn', nbProlfile=20)
generator.visuTest(typePlot='randomPoly', className='Corn')
```
#### Evaluation visualisation
```python
from EvalAlgo.EvalAlgo import *

pathTwoClass = './data/TwoClass/'
pathFiveClass = './data/FiveClass/'
pathTenClass = './data/TenClass/'

systematicChange = False

visualisationEval(pathTwoClass, 'Two classes', systematicChange)
```
## Results
#### Generating data
![Plots Results Generating Data](/img/plotsResults.png)
#### Evaluation
![Plots Results Evaluation](/img/results2_5_10Class.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## Contributors
 - Martin Dautriche - undergraduate computer science student at Univ. Bretagne Sud
 - [Dr. Charlotte Pelletier](https://sites.google.com/site/charpelletier) - Ass. Professor in computer science at Univ. Bretagne Sud / IRISA
 
## License
[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)
