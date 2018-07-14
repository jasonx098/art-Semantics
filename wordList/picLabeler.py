import os
from collections import OrderedDict
import re
from nltk.stem import WordNetLemmatizer as wnl
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import enchant

class TitlePath:
	def __init__(self, path, titleList, indices):
		self.path = path
		self.titleList = titleList
		self.indices = indices
	def __repr__(self):
		return path + "," + titleList + "," + indices

# given a word that is WN tagged and lemmatizer
# lemmatize it and return simplified string with original tag
def lemmatizer(wnTag, wnl):
	if wnTag[1] == 'none':
		return (wnl.lemmatize(wnTag[0]), wnTag[1])
	else:
		return (wnl.lemmatize(wnTag[0], wnTag[1]), wnTag[1])

# given an array of words that represent a string,
# clean the string of whitespace/empty strings and tag
def tagger(strArray):
	# delete all extra white space in array
	cleanArray = []
	for word in strArray:
		if word.strip():
			cleanArray.append(word.strip())

	return pos_tag(cleanArray)


# return a new tuple of wordnet tag given a tuple
# with penn tag
def PennToWN(tuple):
	# noun => 'n'
	if tuple[1][0] == "P" or tuple[1][0] == "N":
		return (tuple[0], 'n')

	# adj => 'a'
	elif tuple[1][0] == "J":
		return (tuple[0], 'a')

	# verb => 'v'
	elif tuple[1][0] == "V":
		return (tuple[0], 'v')

	# adverb => 'r'
	elif tuple[1][0:2] == "RB":
		return (tuple[0], 'r')

	elif tuple[1] in ["DT", "IN", "TO", "CC"]:
		return (tuple[0], 'x')

	else:
		return (tuple[0], 'none')

# synonym synsets given tag or not
def syns(wordTag):
	synonyms = []
	if wordTag[1] not in ['n', 'a', 'v', 'r']:
		for ss in wn.synsets(wordTag[0]):
			for syn in ss.lemma_names():
				synonyms.append(syn)
	else:
		for ss in wn.synsets(wordTag[0], wordTag[1]):
			for syn in ss.lemma_names():
				synonyms.append(syn)
	return synonyms

# trims the file into trimmed title with tuples
def trimmer(file):
	# removing the .jpg ending and number in beginning
	title = (file.split(".")[0]).split("_")[1]

	# list of each individual word in title, cleaned
	titleWords = title.split("-")
	cleanTitle = []
	for word in titleWords:
		cleanTitle.append(re.sub("\(.*\)", "", word))
	# tagged title
	strWithTags = tagger(cleanTitle)

	lemmTitle = []
	for wordTagged in strWithTags:
		if d.check(wordTagged[0]) and wordTagged[0].isalpha():
			tagWN = PennToWN(wordTagged)

			# removing prepositions and one letter words
			if tagWN[1] == "x" or len(tagWN[0]) == 1:
				continue

			# simplified word in a tuple
			simplified = lemmatizer(tagWN, wnl)
			lemmTitle.append(simplified)
	return lemmTitle


# generate the list of words
f = open('word_freq_final.txt', 'r')
wordList = []
count = 0
for line in f:
	wordList.append(line.split(',')[0])
wordList.sort()
"""g = open('wordlist_final.txt', 'w')
for i in range(len(wordList)):
	g.write(str(i) + "-" + wordList[i] + "\n")"""

cwd = os.getcwd()
valid_ext = ".jpg"

d = enchant.Dict("en")
wnl = wnl()

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

tpList = []

for relpath in paths:
	path = cwd + "/" + relpath

	for file in os.listdir(path):
		# if extension is not .jpg
		if os.path.splitext(file)[1].lower() != valid_ext:
			continue

		lemmTitle = trimmer(file)

		indices = []
		titleList = []

		for tup in lemmTitle:
			if tup[0] in wordList:
				indices.append(wordList.index(tup[0]))
				titleList.append(tup[0])
			else:
				for i in syns(tup):
					if i in wordList:
						indicies.append(wordList.index(i))
						titleList.append(i)
						break
		if len(indices) == 0 or len(titleList) == 0:
			continue
		finalIndex = set(indices)
		finalTitle = set(titleList)
		tpList.append(TitlePath(file, set(titleList), set(indices))

a = open("index.txt", "w")

for ti in tpList:
	a.write(ti.file + ",")
	for i in ti.titleList:
		a.write(i + " ")
	a.write(",")
	for j in ti.indices:
		a.write(j + " ")
	a.write(",\n")

