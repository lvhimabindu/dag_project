import numpy as np
import sys
import csv
import re

''' reading and writing files'''

# Helper file supporting reading and writing of files in various formats: list, dictionary, csv -- assumes delimiter is ,

# writes a list of words in file line after line
def writeListInFile(writeLines, filepath):
	with open(filepath, 'w') as f:
		for line in writeLines:
			f.write(line+"\n")

# read a file and fill a list
def readListfromFile(filepath):
	readLines = []
	with open(filepath, 'r') as f:
		readLines = f.read().splitlines()
	return readLines

# write a list of dictionaries with common 'headers' in a file
def writeDictInFile(writeDictList, filepath):

	# expects input which is a list of dictionaries with the same keys shared between all the elements in the list
	if len(writeDictList) == 0:
		print ("Dictionary List is Empty!")
		return	# nothing to do here if the dictionary is empty

	headernames = writeDictList[0].keys()	# keys of the first dictionary will be common to all the other dictionaries in the list, so these are our headers 

	with open(filepath, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=headernames, delimiter=',')
		writer.writeheader()
		writer.writerows(writeDictList)

# read a list of dictionaries from a file
def readDictFromFile(filepath):
	# list of dictionaries that will be read from the file are stored in this 
	readDictList = []
	headernames = []

	with open(filepath,'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = ",")
		
		for row in reader:
			readDictList.append(row)

		# headernames will automatically be read from the first row in the csv file and assigned as the appropriate keys
		if len(readDictList) != 0:
			headernames = list(readDictList[0].keys())

	return headernames, readDictList

# read a list of lines in the csv file and also the header
def readCSV(filepath):
	readLines = []
	headernames = ""
	with open(filepath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter = ",")
		headernames = next(reader)
		for row in reader:
			readLines.append(row)
	return headernames, readLines

# given a header and a list of lines, write a csvfile 
def writeCSV(writeLines, headernames, filepath):
	with open(filepath, 'w', newline = '') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(headernames)
		for row in writeLines:
			writer.writerow(row)

'''if __name__ == '__main__':

	headers1, CSV1 = readCSV("../rt-polaritydata/fulldata_randomized.csv")
	print (headers1)
	print (CSV1)
	writeCSV(CSV1, headers1, "../rt-polaritydata/test-writecsv.csv")

	headers2, Dict2 = readDictFromFile("../rt-polaritydata/fulldata_randomized.csv")
	print (headers2)
	print (Dict2)
	writeDictInFile(Dict2, "../rt-polaritydata/test-writedict.csv")

	List3 = readListfromFile("../rt-polaritydata/fulldata_randomized.csv")
	print (List3)
	writeListInFile(List3, "../rt-polaritydata/test-writelines.csv")'''
	