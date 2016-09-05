import sys
import numpy as np
import random

class Algorithm(object):

	''' Class where we put all the algorithms and also find the best one given a dag and number of machines '''

	def __init__(self, num_machines):
		self.num_machines = num_machines

	def algo_1(self, adj_mat, job_time):
		''' Jana: Fill this part '''
		return random.randint(1, 10)

	def algo_2(self, adj_mat, job_time):
		''' Jana: Fill this part '''
		return random.randint(1, 10)

	def findBestAlgo(self, adj_mat, job_time):
		time_a1 = self.algo_1(adj_mat, job_time)
		time_a2 = self.algo_2(adj_mat, job_time)
		if time_a1 <= time_a2:
			return 0				# 0 indicates algorithm 1; this is the best if its time is the lowest
		else:
			return 1				# 1 indicates algorithm 2

	

''' main starts below, explaining interface to the class '''
'''adj_mat = np.zeros((3,3), dtype=np.int)
adj_mat[0,2] = 1
adj_mat[2,1] = 1
job_time = [3, 5, 5]
alg_obj = Algorithm(2)
print (alg_obj.findBestAlgo(adj_mat, job_time))'''
