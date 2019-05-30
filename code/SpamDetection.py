from nltk import stem
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from pandas import DataFrame
import re
import math
import matplotlib.pyplot as plt

stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

class SpamDetector(object):
	def __init__(self):
		super(SpamDetector, self).__init__()
		messages=pd.read_csv('train_test.csv')
		# divide data two train and test
		self.lenProbSpam = {}
		self.normalMessages = []
		self.vocabFreqSpam = {}
		self.vocabFreqHam = {}
		self.lenProbHam = {}
		self.res = []
		self.spamall = 0
		self.hamall = 0
		self.linkGivenSpam = 0
		self.linkGivenHam = 0
		self.phoneGivenSpam = 0
		self.phoneGivenHam = 0
		
		self.types = messages['type']
		self.texts = messages['text']
		# self.textsTrain = self.texts[:int((len(messages)*8)/10)+1]
		# self.textsTest = list(self.texts[int((len(messages)*8)/10):])
		self.textsTrain = self.texts[:]
		self.textsTest = list(self.texts[:])
		for x in messages['text']:
			self.normalMessages.append(self.alternative_review_messages(x))
		
		self.trainData = self.normalMessages[:]
		self.testData = self.normalMessages[:]
		self.trainDataTypes = self.types[:]
		self.testDataTypes = self.types[:]
		# self.trainData = self.normalMessages[:int((len(messages)*8)/10)+1]
		# self.testData = self.normalMessages[int((len(messages)*8)/10): ]
		# self.trainDataTypes = self.types[:int((len(messages)*8)/10)+1]
		# self.testDataTypes = self.types[int((len(messages)*8)/10): ]
		# print([x for x in self.testDataTypes])
		self.lenEvidence()
		self.vocabEvidence()
		self.linkEvidence()
		self.phoneEvidence()

	def showHist(self, data, name):
		n, bins, patches = plt.hist(data, color='g')
		plt.xlabel(name)
		plt.ylabel('Frequency')
		plt.title(str('Histogram of ') + name)
		plt.grid(True)
		plt.show()

	def ploting(self):
		spamLens = []
		hamLens = []
		i = 0
		for x in self.texts:
			if self.types[i] == 'spam':
				spamLens.append(len(x))
			else:
				hamLens.append(len(x))
			i += 1
		self.showHist(spamLens, 'spam\'s len')
		self.showHist(hamLens, 'ham\'s len')

	def runOnTestCase(self, path):
		messages=pd.read_csv(path)
		# print(messages)
		self.testData = []
		texts = messages['text']
		self.textsTest = list(texts[:])
		ids = list(messages['id'])
		for x in messages['text']:
			self.testData.append(self.alternative_review_messages(x))
		self.evaluateTest()
		# print(self.res)
		result = {'id':ids, 'type':self.res}
		df = DataFrame(result, columns= ['id', 'type'])
		export_csv = df.to_csv (r'output.csv', index = None, header=True)
		# print(df)


	def evaluateTest(self):
		pspam = float(self.spamall)/(self.spamall+self.hamall)
		pham = float(self.hamall)/(self.spamall+self.hamall)
		self.res = []
		spamallwords = 0
		hamallwords = 0
		i = 0
		for x in self.vocabFreqSpam.keys():
			spamallwords += self.vocabFreqSpam[x]
		for x in self.vocabFreqHam.keys():
			hamallwords += self.vocabFreqHam[x]
		for x in self.testData:
			spamMail = math.log(pspam)
			hamMail = math.log(pham)
			if len(self.textsTest[i]) in self.lenProbSpam.keys():
				spamMail += math.log(self.lenProbSpam[len(self.textsTest[i])])
			else:
				spamMail += math.log(0.0000005/spamallwords)
			if len(self.textsTest[i]) in self.lenProbHam.keys():
				hamMail += math.log(self.lenProbHam[len(self.textsTest[i])])
			else:
				hamMail += math.log(0.0000005/spamallwords)
			urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)
			if len(urls) > 0:
				spamMail += math.log(self.linkGivenSpam)
				hamMail += math.log(self.linkGivenHam)
			phones = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', x)
			if len(phones) > 0:
				spamMail += math.log(self.phoneGivenSpam)
				hamMail += math.log(self.phoneGivenHam)
			words = x.split()
			for y in words:
				if y in self.vocabFreqSpam.keys():
					spamMail += math.log((float(self.vocabFreqSpam[y])/spamallwords))
				else:
					spamMail += math.log(0.000005/spamallwords)
				if y in self.vocabFreqHam.keys():
					hamMail += math.log((float(self.vocabFreqHam[y])/hamallwords))
				else:
					hamMail += math.log(0.000005/hamallwords)
			if spamMail > hamMail:
				self.res.append('spam')
			else:
				self.res.append('ham')
			i += 1


	def test(self):
		self.evaluateTest()
		self.showResult(self.res)
		self.testData = self.normalMessages[:int((len(self.normalMessages)*8)/10)+1]
		self.testDataTypes = self.types[:int((len(self.normalMessages)*8)/10)+1]
		self.textsTest = list(self.texts[:int((len(self.normalMessages)*8)/10)+1])
		self.evaluateTest()
		self.showResult(self.res)


	def showResult(self, res):
		correctDetectedSpam = 0
		detectedSpam = 0
		correctDetected = 0
		totalSpam = 0
		total = 0
		testDataTypes = [x for x in self.testDataTypes]
		i = 0
		for x in res:
			if x == 'spam':
				detectedSpam += 1
				if testDataTypes[i] == 'spam':
					correctDetectedSpam += 1
					correctDetected += 1
			else:
				if testDataTypes[i] == 'ham':
					correctDetected += 1
			i += 1
		for x in testDataTypes:
			if x == 'spam':
				totalSpam += 1
			total += 1

		print("recall = " + str(float(correctDetectedSpam)/totalSpam) + " precision = " + str(float(correctDetectedSpam)/detectedSpam) + " accuracy = " + str(float(correctDetected)/total))

	def linkEvidence(self):
		i = 0
		spamGivenLink = 0
		hamGivenLink = 0
		for x in self.trainData:
			urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)
			# print(urls)
			if self.trainDataTypes[i] == 'spam' and len(urls) > 0:
				spamGivenLink += 1
			elif self.trainDataTypes[i] == 'ham' and len(urls) > 0:
				hamGivenLink += 1
			i += 1
		if spamGivenLink == 0:
			spamGivenLink += 0.00005 
		self.linkGivenSpam = float(spamGivenLink)/self.spamall
		if hamGivenLink == 0:
			hamGivenLink += 0.00005 
		self.linkGivenHam = float(hamGivenLink)/self.hamall

	def phoneEvidence(self):
		i = 0
		spamGivenPhone = 0
		hamGivenPhone = 0
		for x in self.trainData:
			phones = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', x)
			# print(phones)
			if self.trainDataTypes[i] == 'spam' and len(phones) > 0:
				spamGivenPhone += 1
			elif self.trainDataTypes[i] == 'ham' and len(phones) > 0:
				hamGivenPhone += 1
			i += 1
		if spamGivenPhone == 0:
			spamGivenPhone += 0.00005 
		self.phoneGivenSpam = float(spamGivenPhone)/self.spamall
		if hamGivenPhone == 0:
			hamGivenPhone += 0.00005 
		self.phoneGivenHam = float(hamGivenPhone)/self.hamall

	def vocabEvidence(self):
		i = 0
		for x in self.trainData:
			words = x.split()
			# words = list(dict.fromkeys(words))
			for y in words:
				if self.trainDataTypes[i] == 'spam':
					if not y in self.vocabFreqSpam.keys():
						self.vocabFreqSpam[y] = 1
					else:
						self.vocabFreqSpam[y] += 1
				else:
					if not y in self.vocabFreqHam.keys():
						self.vocabFreqHam[y] = 1
					else:
						self.vocabFreqHam[y] += 1
			i += 1

		# print(self.vocabFreqSpam)
		# print(self.vocabFreqHam)

	def lenEvidence(self):
		hamall = 0
		spamall = 0
		for x in range(0, len(self.textsTrain)):
			if self.trainDataTypes[x] == 'spam':
				spamall += 1
			else:
				hamall += 1
		# print((spamall, hamall))
		maxLen = len(max(self.textsTrain, key=len))
		# print(maxLen)
		for l in range(1,maxLen):
			lenGivenSpam = 0
			lenGivenHam = 0
			for x in range(0, len(self.trainDataTypes)):
				if self.trainDataTypes[x] == 'spam' and len(self.textsTrain[x]) == l:
					lenGivenSpam += 1
				elif self.trainDataTypes[x] == 'ham' and len(self.textsTrain[x]) == l:
					lenGivenHam += 1
			#math.log
			if lenGivenSpam == 0:
				lenGivenSpam += 0.00005
			if lenGivenHam == 0:
				lenGivenHam += 0.00005
			self.lenProbSpam[l] = lenGivenSpam/spamall
			self.lenProbHam[l] = lenGivenHam/hamall
		self.spamall = spamall
		self.hamall = hamall
		# print(self.lenProbHam)
		# print(self.lenProbSpam)

	def alternative_review_messages(self, msg):
		stemmer = SnowballStemmer("english")
		msg = msg.replace(",", " ")
		msg = msg.replace(".", " ")
		msg = msg.replace("?", " ")
		msg = msg.replace("!", " ")
		# converting messages to lowercase
		msg = msg.lower()
		# removing stopwords
		msg = [word for word in msg.split() if word not in stopwords]
		# using a stemmer
		msg = " ".join([stemmer.stem(word) for word in msg])
		# print(msg)
		# print(stemmer.stem("girls"))
		return msg

sd = SpamDetector()
# sd.test()
sd.runOnTestCase('evaluate.csv')
# sd.ploting()