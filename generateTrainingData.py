import sys
import numpy as np
import random

from my_exceptions import MyError
from topologicalsort import TopoSort

''' 
Class designed to fix a graph to be acyclic (DAG) as well as return topological ordering of the graph
'''

class TrainingData(object):

	def __init__(self, min_num_jobs, max_num_jobs, min_job_time, max_job_time, min_prob_edge, max_prob_edge, num_machines, num_data_points):
		
		# min and max number of jobs
		self.min_num_jobs = min_num_jobs
		self.max_num_jobs = max_num_jobs

		# min and max job time
		self.min_job_time = min_job_time
		self.max_job_time = max_job_time

		# min and max edge formation probability in the graph -- note that in the code below, we remove the edges that create cycles, so the probability might go a little above/below the limits
		self.min_prob_edge = min_prob_edge
		self.max_prob_edge = max_prob_edge

		# number of dags (datapoints)
		self.num_data_points = num_data_points

		# number of machines 
		self.num_machines = num_machines

		# for each dag (datapoint), generate the number of jobs and the time taken by each of those jobs
		self.num_jobs_arr = [random.randint(min_num_jobs, max_num_jobs) for i in range(num_data_points)]
		self.job_time_arr = [[random.randint(min_job_time, max_job_time) for j in range(self.num_jobs_arr[i])] for i in range(num_data_points)]

		# initializing the training_data array 
		self.training_data = []

	def genDatapointsLabels(self, filepath):
		
		algo_obj = Algorithm(self.num_machines)

		for i in range(num_data_points):

			# randomly pick the probability of edge formation for this particular graph 
			prob_edge = random.uniform(self.min_prob_edge, self.max_prob_edge)

			# generate a dag and get the topologically sorted list of nodes
			mat, topo_sorted_list = self.genAdjMatrix(prob_edge, self.num_jobs_arr[i])

			# find the best algorithm for this particular dag 
			best_alg = algo_obj.findBestAlgo(mat, self.job_time_arr[i])

			# generate feature vector - min., max., avg. in degree; min, max, avg. out degree; min path length, max path length, max. job time, min. job time, avg. job time
			features = self.genFeatVector(mat, self.num_jobs_arr[i], self.job_time_arr[i], topo_sorted_list)

			# append the newly created feature vector and the best alg. index to the training data
			self.training_data.append(features+[best_alg])

		writeCSV(self.training_data, ['min. in deg', 'max. in deg', 'min. out deg', 'max. out deg', 'avg. deg', 'min job time', 'max job time', 'avg. job time', 'min path', 'max path', 'label'], filepath)


	def genFeatVector(self, mat, num_jobs, job_time_list, topo_sorted_list):

		# compute min, max, avg. in and out degree
		max_in_degree = 0
		min_in_degree = num_jobs
		sum_in_degree = 0

		for i in range(num_jobs):
			temp_in_degree = 0
			for j in range(num_jobs):
				temp_in_degree += mat[j,i]
			max_in_degree = max(temp_in_degree, max_in_degree)
			min_in_degree = min(temp_in_degree, min_in_degree)
			sum_in_degree += temp_in_degree

		avg_degree = (sum_in_degree+0.0)/num_jobs

		max_out_degree = 0
		min_out_degree = num_jobs
		
		for i in range(num_jobs):
			temp_out_degree = 0
			for j in range(num_jobs):
				temp_out_degree += mat[i,j]
			max_out_degree = max(temp_out_degree, max_out_degree)
			min_out_degree = min(temp_out_degree, min_out_degree)

		# compute min, max, avg. job time
		min_job_time = min(job_time_list)
		max_job_time = max(job_time_list)
		avg_job_time = (sum(job_time_list)+0.0)/num_jobs

		# compute min., max path
		max_path = 0
		min_path = max_job_time * num_jobs

		# use the topo sorted list to identify the longest and shortest paths
		def getParents(self, s):
		parents = []
		for i in range(self.N):
				if self.mat[i, s] == 1:
					parents.append(i)
		return parents

		return [min_in_degree, max_in_degree, min_out_degree, max_out_degree, avg_degree, min_job_time, max_job_time, avg_job_time, min_path_job_length, max_path_job_length]


	def genAdjMatrix(self, prob_edge, num_jobs):
		
		# initialize the adjacency matrix with zeros
		mat = np.zeros((num_jobs, num_jobs), dtype=np.int)

		# generate dependencies between the jobs 
		for i in range(num_jobs):
			for j in range(num_jobs):
				p = random.uniform(self.min_prob_edge, self.max_prob_edge)
				if prob_edge <= p:
					mat[i,j] = 1

		# fix the dependencies such that the dag structure is preserved and return the resulting dag
		return self.checkFixDag(mat, num_jobs)

	def checkFixDag(self, mat, num_jobs): 

		topo_obj = TopoSort(mat, num_jobs)

		return topo_obj.getDAG(), topo_obj.getTopoSorted()
		

''' MAIN CODE TO TEST THE CLASS '''

train_data_obj = TrainingData(10, 2, 5, 1000)
