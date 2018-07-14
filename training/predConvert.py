f = open('all_ind.txt', 'r')
g = open('wordlist_all.txt', 'r')

predictions = []
words = []
for line in g:
	words.append(line.split('-')[1].rstrip())
g.close()


for line in f:
	titleWords = []
	for index in line.strip().split(' '):
		if index:
			titleWords.append(words[int(index)])
	predictions.append(titleWords)
f.close()

h = open('all_words.txt', 'w')
for eachTest in predictions:
	for word in eachTest:
		h.write(word + ",")
	h.write('\n')

h.close()