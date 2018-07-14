# creates a text file where all the titles are present
# separated by commas

from collections import OrderedDict
import os
import re
from nltk.corpus import wordnet as wn

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

titles = open('title_list.txt', 'w')

# all paths in data set
for relpath in paths:

	path = cwd + "/" + relpath

	# every file in the list
	flist = os.listdir(path)
	for file in flist:
		# if extension is not .jpg
		if os.path.splitext(file)[1].lower() != valid_ext:
			continue
		# removing the .jpg ending and number in beginning
		title = (file.split(".")[0]).split("_")[1]

		# list of each individual word, split by whatever
		titleWords = title.split("-")
		for w in titleWords:
			w = re.sub("\(.*\)", "", w)
			titles.write(w + ',')
		titles.write("\n")


