import os
import shutil

f = open('test_all.txt', 'r')
vals = open('all_val.txt', 'r')
words = open('all_words.txt', 'r')
cwd = os.getcwd()
target = "/home/jason/Documents/Jason/training1/imgNotes/"

for line in f:
	path = line.split(',')[0]
	name = path.split('/')[-1]
	category = path.split('/')[-2]

	#targetPath = target + category + "/" + name
	#shutil.copyfile(path, targetPath)

	values = vals.readline().rstrip('\n').split(' ')
	wordList = words.readline().rstrip('\n').split(',')

	title = name.split('.')[0]
	textPath = target + category + "/annotation/" + title + ".txt"
	textFile = open(textPath, 'w+')
	
	for i in range(2506):
		textFile.write(wordList[i] + '(' + values[i] + ') ')

