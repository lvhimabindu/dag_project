import sys
import numpy as np
import random

from generateTrainingData import TrainingData
from my_exceptions import MyError
from topologicalsort import TopoSort
from Algorithm import Algorithm
from readwritefiles import writeCSV
from classification import classification

train_data_obj = TrainingData(3, 8, 2, 10, 0.7, 1.0, 5, 100)
train_data_obj.genDatapointsLabels("temp.csv")
class_obj = classification("Random Forest")
print(class_obj.getCVAUC("temp.csv"))