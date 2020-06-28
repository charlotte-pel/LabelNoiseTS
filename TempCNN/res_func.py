#!/usr/bin/python

"""
	Saving the resultss
"""

import os, sys
import argparse

import numpy as np
import pandas as pd
import math
import random
import itertools
import time


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------			SAVE RESULTS			--------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------		
def saveLossAcc(model_hist, filename):
	""" 
		Save all the accuracy measures into a csv file
		INPUT:
			- model_hist: all accuracy measures
			- filename: csv file where to store the dictionary 
				(erase the file it does already exist)
				8 significant digits after the decimal point
	"""
	f = open(filename, 'w')
	for key in model_hist.keys():
		line = key + ',' + ','.join(map(str, model_hist[key])) + '\n'
		f.write(line)
	f.close()

#-----------------------------------------------------------------------		
def saveMatrix(mat, filename, label):
	""" 
		Save numpy array into a csv file
		INPUT:
			- mat: numpy array
			- filename: csv file where to store the mat array 
				(erase the file it does already exist)
				8 significant digits after the decimal point
			- label: name of the columns
	"""
	df = pd.DataFrame(mat, columns=label)
	df.to_csv(filename)
	
#-----------------------------------------------------------------------
def save_confusion_matrix(C, class_name, conf_file):
	""" 
		Create a confusion matrix with IndexName, Precision, Recall, F-Score, OA and Kappa
		Charlotte's style
		INPUT:
			- C: confusion_matrix compute by sklearn.metrics.confusion_matrix
			- class_name: corresponding name class
		OUTPUT:
			- conf_mat: Charlotte's confusion matrix
	"""
	
	nclass, _ = C.shape
	
	#-- Compute the different statistics
	recall = np.zeros(nclass)
	precision = np.zeros(nclass)
	fscore = np.zeros(nclass)
	diag_sum = 0
	hdiag_sum = 0
	for add in range(nclass):
		hdiag_sum = hdiag_sum + np.sum(C[add,:])*np.sum(C[:,add])
		if C[add,add] == 0:
			recall[add] =0
			precision[add] =0
			fscore[add] =0
		else:
			recall[add] = C[add,add]/np.sum(C[add,:])
			recall[add] = "%.6f" % recall[add]
			precision[add] = C[add,add]/np.sum(C[:,add])
			precision[add] = "%.6f" % precision[add]
			fscore[add] = (2*precision[add]*recall[add])/(precision[add]+recall[add])
			fscore[add] = "%.6f" % fscore[add]
	nbSamples = np.sum(C)
	OA = np.trace(C)/nbSamples
	ph = hdiag_sum/(nbSamples*nbSamples)
	kappa = (OA-ph)/(1.0-ph)
			
	f = open(conf_file, 'w')
	line = ' '
	for name in class_name:
		line = line + ',' + name
	line = line + ',Recall\n'
	f.write(line)
	for j in range(nclass):
		line = class_name[j]
		for i in range(nclass):
			line = line + ',' + str(C[j,i])
		line = line + ',' + str(recall[j]) + '\n'
		f.write(line)
	line = "Precision"
	for add in range(nclass):
		line = line + ',' + str(precision[add])
	line = line + ',' + str(OA)
	line = line + ',' + str(kappa) + '\n'
	f.write(line)
	line = "F-Score"
	for add in range(nclass):
		line = line + ',' + str(fscore[add])
	line = line + '\n'
	f.write(line)
	f.close()
	
#-----------------------------------------------------------------------		
def computingConfMatrix(referenced, p_test, n_classes):
	""" 
		Computing a n_classes by n_classes confusion matrix
		INPUT:
			- referenced: reference data labels
			- p_test: predicted 'probabilities' from the model for the test instances
			- n_classes: number of classes (numbered from 0 to 1)
		OUTPUT:
			- C: computed confusion matrix
	"""
	predicted = p_test.argmax(axis=1)
	C = np.zeros((n_classes, n_classes))
	for act, pred in zip(referenced, predicted):
		C[act][pred] += 1
	return C
			
#EOF
