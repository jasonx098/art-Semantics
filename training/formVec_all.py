import numpy as np
from scipy.misc import imread, imresize
import os

cwd = os.getcwd()
valid_ext = ".jpg"
# files of data set images
# the data set must be contained in a folder (titled "imgs") within the working directory
paths = ["imgs/1_early-renaissance"
		, "imgs/2_high-renaissance"
		, "imgs/3_mannerism-late-renaissance"
		, "imgs/4_northern-renaissance"
		, "imgs/5_baroque"
		, "imgs/6_rococo"
		, "imgs/7_romanticism"
		, "imgs/8_impressionism"
		, "imgs/9_post-impressionism"
		, "imgs/10_realism"
		, "imgs/11_art-nouveau-modern"
		, "imgs/12_cubism"
		, "imgs/13_expressionism"
		, "imgs/14_fauvism"
		, "imgs/15_abstract-expressionism"
		, "imgs/16_color-field-painting"
		, "imgs/17_minimalism"
		, "imgs/18_na-ve-art-primitivism"
		, "imgs/19_ukiyo-e"
		, "imgs/20_pop-art"
		]

f = open('index_all.txt', 'r')

# from dictionary with file name and indices
titleIndex = {}
for line in f:
	intIndex = []
	for i in line.split(',')[2].strip().split(' '):
		intIndex.append(int(i))
	titleIndex[line.split(',')[0]] = intIndex

titles = titleIndex.keys()

fullPathIndex = {}
for relpath in paths:
	path = cwd + "/" + relpath

	for file in os.listdir(path):
		if file in titles:
			fullpath = os.path.join(path, file)
			currimg = imread(fullpath)

			# weird exceptions
			if len(currimg.shape) != 3:
				continue
			if currimg.shape[2] != 3:
				continue
			
			# dictionary of fullpath and a list of indices
			fullPathIndex[fullpath] = titleIndex[file]

count = len(fullPathIndex)

# randomnly shuffling a vector list to determine training + testing
# testing 20%, training 80%
randidx = np.arange(count)
np.random.shuffle(randidx)

trainidx = randidx[0:int(4 * count / 5)]
testidx = randidx[int(4 * count / 5): count]

g = open('training_all.txt', 'w')
h = open('testing_all.txt', 'w')

for tup in enumerate(fullPathIndex.items()):
	if tup[0] in trainidx:
		g.write(tup[1][0] + ",")
		for index in tup[1][1]:
			g.write(str(index) + " ")
		g.write("\n")
	if tup[0] in testidx:
		h.write(tup[1][0] + ",")
		for index in tup[1][1]:
			h.write(str(index) + " ")
		h.write("\n")
	
g.close()
h.close()
	




