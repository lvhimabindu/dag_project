import sys

''' Exception class '''

class MyError(Exception):

	def __init__(self, value):
		self.value = value

	def __str__(self):
		return "Exception raised: %s " % (self.value)