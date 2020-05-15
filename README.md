# Generating synthetic time series dataset to study the influence of class label noise on classification performance

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
- Matplotlib (Only for visualisation)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas.

```bash
pip install numpy
pip install pandas
pip install Matplotlib # (Only for visualisation)
```

## Quick use

Open and run the notebook: ExampleUse in jupyter notebook.\
Located at the root of the project folder

## Usage

```python
from SyntSITSLabelNoise.GeneratorData import *

generator = GeneratorData("src/dataFrame.h5")
a = {'Wheat': 'Barley', 'Barley': 'Soy'}
(X,Y) = generator.getDataXY() 
(Xnoise,YNoise) = generator.getNoiseDataXY(0.05,a) #Example with systematic change label noise
(Xtest,Ytest) = (generator.getTestData())
```
## Config File:

DataFrame from initFile.csv must be like below :
(Example initFile.csv has good format)
- First column must be named: class_names
- Last row must contain dates (In the example below line 13), dates start column after class_names column.
```
      class_names     0       1      2      3  ...    21     22     23    24    25
0            Corn  0.57   0.720   0.15   0.30  ...  -1.0   -1.0   -1.0  -1.0  -1.0
1   Corn_Ensilage  0.57   0.720   0.15   0.30  ...  -1.0   -1.0   -1.0  -1.0  -1.0
2         Sorghum  0.62   0.770   0.15   0.30  ...  -1.0   -1.0   -1.0  -1.0  -1.0
3       Sunflower  0.67   0.820   0.15   0.30  ...  -1.0   -1.0   -1.0  -1.0  -1.0
4             Soy  0.67   0.820   0.15   0.30  ...  -1.0   -1.0   -1.0  -1.0  -1.0
5           Wheat  0.52   0.670   0.20   0.35  ...  -1.0   -1.0   -1.0  -1.0  -1.0
6        Rapeseed  0.70   0.800   0.05   0.20  ...  12.0  135.0  145.0   5.0  15.0
7          Barley  0.52   0.670   0.20   0.35  ...  -1.0   -1.0   -1.0  -1.0  -1.0
8       Wheat_Soy  0.50   0.550   0.10   0.15  ...  15.0  280.0  300.0  25.0  35.0
9       Evergreen  0.01   0.015   0.55   0.70  ...  -1.0   -1.0   -1.0  -1.0  -1.0
10      Decideous  0.20   0.350   0.40   0.50  ...  -1.0   -1.0   -1.0  -1.0  -1.0
11          Water  0.01   0.020  -0.20   0.00  ...  -1.0   -1.0   -1.0  -1.0  -1.0
12          Build  0.01   0.020   0.20   0.30  ...  -1.0   -1.0   -1.0  -1.0  -1.0
13            NaN  1.00  26.000  51.00  76.00  ...   NaN    NaN    NaN   NaN   NaN
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## Contributors
 - Martin Dautriche - undergraduate computer science student at Univ. Bretagne Sud
 - [Dr. Charlotte Pelletier](https://sites.google.com/site/charpelletier) - Ass. Professor in computer science at Univ. Bretagne Sud / IRISA
 
## License
[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)
