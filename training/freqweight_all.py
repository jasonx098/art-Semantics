import os
from collections import OrderedDict

def calculateWeights():
	f = open('word_freq_all.txt', 'r')
	freq = {}
	totalSum = 0
	for line in f:
		stringLine = line.split('\n')[0].split(",")
		freq[stringLine[0]] = int(stringLine[1])
		totalSum += int(stringLine[1])
	sortFreq = OrderedDict(sorted(freq.items(), key = lambda t:t[1], reverse = True))

	weightedList = []
	for key, val in sortFreq.items():
		# use 100 or whatever normalizing value?
		weightedList.append(float(totalSum) / (val * 100))

	return weightedList

