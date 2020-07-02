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
- h5py (HDF5 for Python)
- Jupyter Notebook (only for use Jupyter Notebook)
- Matplotlib (only for visualisation)
- Scikit-Learn, use to train traditional machine learning algorithms including Support Vector Machines (SVM) and Random Forest (RF)
- Keras & Tensorflow, use to train deep learning architectures such as Temporal Convolutional Neural Network (TempCNN)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas.

```bash
pip install numpy
pip install pandas
pip install tables
pip install h5py
pip install jupyter 
pip install matplotlib
pip install sklearn
pip install keras==2.4.3 tensorflow==2.2.0 
```

## Quick use
There are a set of Jupyter notebooks available in the `notebook` folder, for example:
* `ExampleUseGeneratingData.ipynb` generates data...
* `ExampleUseEvaluation.ipynb` trains the following models on...

## Data generation
The following Python code is used to generate a 2-class dataset (wheat and barley) composed of the original data `(X,Y)`, the original data corrupted by 5 % of random class label noise `(Xnoise,Ynoise)`, and some non noisy test data (`(Xtest,Ytest)`. `X` (of size `(n,l)`) is the data matrix (numpy array) composed of `n` time series of length `l` (`X[i,j]` represents the NDVI value of the `i` observation at time `j`), and `Y` is the label vector (of size `n`) associated to each time series.
```python
from GenLabelNoiseTS.GenLabelNoiseTS import *

path = './somePath/'
# Example with a list of two class and systematic change.
generator = GenLabelNoiseTS(filename="dataset.h5", classList=('Wheat','Barley'), csv=True, verbose=True, dir=path/to/dir)
a = {'Wheat': 'Barley', 'Barley': 'Wheat'}
(X,Y) = generator.getDataXY()
(Xnoise,Ynoise) = generator.getNoiseDataXY(0.05,a)
(Xtest,Ytest) = (generator.getTestData())
generator.defaultVisualisation()
```
#### Generating a dataset using a Python command line
A dataset can be generated with the following Python command line.
```bash
# Generating a non-noisy synthetic dataset composed of original data and some corrupted version with a random class label noise applied to each class
python gen_data.py -d path/to/dir -f dataset.h5 -noClass 10 -noise random -noise.level [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis
# Generating a non-noisy synthetic dataset composed of original data and some corrupted version with a systematic class label noise applied to each class
python gen_data.py -d path/to/dir -f dataset.h5 -noClass 10 -noise {'Wheat':('Barley','Soy'),'Barley':'Soy'} -noise.level [0.05,0.1,0.15,0.2,0.25,0.3] -save_csv -v -vis
# The dictionary is used to add a systematic noise to the original data. In this example, the wheat labels are always changed to either barley or soy.
# The dictionay should not contain any space: {'Wheat':('Barley','Soy'),'Barley':'Soy'}
```
This command generates a dataset in the specified directory `path/to/dir`. The complete dataset is stored in the HDF5 `dataset.h5`. It includes non-noisy data as well as noisy data contaminated by different level of noise (`noise.level`).\
The following options are mandatory:
- `-d path/to/dir`: directory path
- `-f fileName`: file name (should be a .h5 file)
- `-noClass 10`: number of class presents in the generated dataset
- `-noise random`: type of class label noise added to the data. Either `random` or a dictionary for systematic noise. The dictionary needs to be in a Python format: {'Wheat':('Barley','Soy'),'Barley':'Soy'}.
- `-noise.level [0.05,0.1]`: a list containing the noise levels added to the data.

The following options are optional:
- `-save_csv`: the data will be also saved in csv files (one file per type of noise and per level of noise)
- `-v`: verbose mode
- `-vis`: saving some default visualisation

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
    
## Configuration file

The configuration file `initFile.csv` is a data frame (converted into csv file) with the following format:
- The first column must be named: `class_names`.
- The `0` and `1` columns contain the number of samples and the number of polygons for each class. The following columns contain parameter values used to parametrized a double logistic function (please refer to the Remote Sensing journal paper).
- The last row (line 13 in the folloing example) must contain days of year (dates). The days of year are inserted from column `0`.
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
The following code retrieves the data contain in `dataset.h5` file and performs several visualisation on the non-noisy data.
```python
from GenLabelNoiseTS.GenLabelNoiseTS import *

generator = GenLabelNoiseTS(filename="dataset.h5", dir='pathToData' + 'Run' + str(1) + '/', csv=True,
                                verbose=False)
generator.visualisation(typePlot='mean')
generator.visualisation(typePlot='mean', className='Corn')
generator.visualisation(typePlot='all', className='Corn')
generator.visualisation(typePlot='random', className='Corn', nbProlfile=20)
generator.visualisation(typePlot='randomPoly', className='Corn')
```

If 'yourPath/' is None, visualisation will not be save in external file.\
If you specify 'yourPath/', visualisation will be save in external file.

Except for visualisationProfilsMeanAllClass, you need specify a class Name besides 'yourPath/'

By default 'yourPath/' is None

#### Algorithm performance visualisation
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
