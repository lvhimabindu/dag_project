import sys
import numpy as np
import random

from my_exceptions import MyError

''' 
Input: Number of machines (Assuming same for all jobs in one training set), Mean & SD of length of each job

Class functions compute:
1. generate adjacency matrix
	- check and fix structure so that it is a dag
3. given the dag adjacency matrix, length of each job, number of machines, find the best algorithm. 
4. generate features of the dag and create a datapoint with the label of the best algorithm. 
5. Writeout a csv file for training data
'''

class TrainingData(object):

	def __init__(self, mean_job_time, sd_job_time, num_machines, num_data_points):
		self.mean_job_length = mean_job_time
		self.sd_job_length = sd_job_time
		self.num_machines = num_machines
		self.num_data_points = num_data_points
		self.num_jobs_arr = np.random.normal(mean_job_time, sd_job_time, num_data_points)
		self.training_data = []

	def genDatapointsLabels(self):
		
		algo_obj = Algorithm(self.num_machines)

		for i in range(num_data_points):
			prob_edge = random.random()
			mat = genAdjMatrix(prob_edge, self.num_jobs_arr[i])
			alg = algo_obj.findBestAlgo(mat)
			features = genFeatVector(mat)
			self.training_data.append(features+[alg])

	def genAdjMatrix(self, prob_edge, num_jobs):
		mat = np.zeros((num_jobs, num_jobs), dtype=np.int)

		return checkFixDag(mat)


	def checkFixDag(self, mat):
		return mat
		
