from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn 
from nltk.stem import WordNetLemmatizer
import enchant
from collections import OrderedDict

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


# read in data from title_list.txt
f = open('title_list.txt', 'r')

# dictionary and lemmatizer
d = enchant.Dict("en")
wnl = WordNetLemmatizer()

# frequency list
freq = {}

# tokenize the sentences first
for line in f:
	# tokenizing the string
	strArray = line.split("\n")[0].split(",")
	strWithTags = tagger(strArray)

	# only use words in dictionary and without numbers
	for wordTagged in strWithTags:
		if d.check(wordTagged[0]) and wordTagged[0].isalpha():
			# changing tag to WN style
			tagWN = PennToWN(wordTagged)

			# removing prepositions and one letter words
			if tagWN[1] == "x" or len(tagWN[0]) == 1:
				continue

			# simplified word
			simplified = lemmatizer(tagWN, wnl)

			# adding into the frequency list
			if simplified in freq:
				freq[simplified] += 1
			else:
				freq[simplified] = 1

# elements with more occurences than six
major = {}
# elements with fewer/equal occurences than six
lessSix = {}
for tupe, num in freq.items():
	if int(num) < 6:
		lessSix[tupe] = int(num)
	else:
		if tupe[0] in major:
			major[tupe[0]] += int(num)
		else:
			major[tupe[0]] = int(num)

for tupe, num in lessSix.items():
	for i in syns(tupe):
		if i in major:
			major[i] += int(num)
			break

sortedFreq = OrderedDict(sorted(major.items(), key = lambda t:t[1], reverse=True))
trim = open('word_freq_final.txt', 'w')
for word in sortedFreq:
	trim.write(word + "," + str(sortedFreq[word]) + "\n")




