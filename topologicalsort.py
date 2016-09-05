import sys
import numpy as np
import random

from my_exceptions import MyError

''' 
Class designed to fix a graph to be acyclic (DAG) as well as return topological ordering of the graph
'''

class TopoSort(object):

	def __init__(self, mat, N):
		self.mat, self.TopoSorted = self.fixCyclesandTopo(mat, N)
		self.N = N

	def fixCyclesandTopo(self, mat, N):
		L = []
		perm_marked = []
		temp_marked = [0] * N

		def visit(s, L, perm_marked, temp_marked):

			if temp_marked[s] == 1:
				return -1

			if s not in perm_marked:
					
				temp_marked[s] = 1
					
				for i in range(N):

					rval = 0
						
					if mat[s,i] == 1:
						rval = visit(i, L, perm_marked, temp_marked)
						
					if rval == -1:
						mat[s,i] = 0		# making the graph a dag 

				temp_marked[s] = 0

				perm_marked.append(s)

				L.insert(0, s)

			return 0

		while len(perm_marked) != N:

			unmarked = set(range(N)) - set(perm_marked)
			s = random.sample(unmarked, 1)[0]

			print (s)

			rval = visit(s, L, perm_marked, temp_marked)

			if rval == - 1:
				raise MyError('something gone wrong horribly!')		

		return mat, L

	def getDAG(self):
		return self.mat

	def getTopoSorted(self):
		return self.TopoSorted


''' MAIN CODE TO TEST THE CLASS '''
mat = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]])
t_obj = TopoSort(mat, 4)
print (t_obj.getDAG())
print (t_obj.getTopoSorted())
#print (t_obj.getParents(3))