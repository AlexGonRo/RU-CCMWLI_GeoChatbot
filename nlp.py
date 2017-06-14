from __future__ import print_function, unicode_literals
import random
import logging
import numpy as np
import word_lists
from utils import seqWords2seqVec, eucDistance
#####


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NLP:
	def __init__(self, data, encoder, word_inxed, embedding_matrix, max_num_vectors, num_features):
		self.data = data # THE PANDA DF
		self.encoder = encoder
		self.word_index = word_inxed
		self.embedding_matrix = embedding_matrix
		self.max_num_vectors = max_num_vectors
		self.num_features = num_features


	def random_answer(self):
		return random.choice(word_lists.GENERAL_RESPONSES)

	def check_for_greeting(self, sentence):
		resp = None
		if any(sentence in s for s in word_lists.GREETING_SET):
			resp = random.choice(word_lists.GREETING_RESPONSES)
		if not resp:
			for word in sentence:
				if word in word_lists.GREETING_SET:
					resp = random.choice(word_lists.GREETING_RESPONSES)
		return resp


	def introduction(self):
		self.askorigin = True
		return word_lists.INTRODUCTION


	def city_in_sentence(self, sentence):
		print ("Look if City in Sentence")
		city_names = self.data['city_name'].tolist()
		city_names_lower = [x.lower() for x in city_names]

		sentence_split = sentence.split()

		for word in sentence_split:
			if word in city_names_lower:
				return True, city_names[city_names_lower.index(word)]
		return False, None

	def describe_city(self, city):
		print ("retrieve Description of city: ".format(city))
		index = self.data.city_name[self.data.city_name == city].index.tolist()[0]
		print (index)
		description = self.data.iloc[index]['description'] # JUST NOW UNTIL WE HAVE THE REAL DF
		description = description + ". Read more about it in https://en.wikivoyage.org/wiki/{}".format(city)
		print ("Decription =")
		print (description)
		return description

#	def propose_alternatives(self, cities):


	# That's the main function that returns our response
	def respond(self, sentence):

		logger.info("Dback: respond to %s", sentence)

		sentence = sentence.lower()
		# Check for introduction
		if sentence=="start" or sentence=="\start":
			resp = self.introduction(); print("Intro2"); print(resp)
			return resp
		# Check for a greeting
		resp = self.check_for_greeting(sentence)
		if resp is not None:
			return resp


		# Check if user is looking for a city description
		city_in_sentence, city = self.city_in_sentence(sentence)
		print("city in sentence ?")
		if city_in_sentence:
			print ("city is in sentence !!!")
			print (city)
			resp = self.describe_city(city)
			return resp

		# I simplified by just assuming that in any other case we give a city suggestion
		# Like this, we don't need Spacy for now
		# If the Input is shorter than x then we simply give a random_answer

		if (len(sentence) < 10):
			resp = self.random_answer()
			return resp

		else:
			indexes = self.match_description(sentence)
			resp = self.suggest_city(indexes)
			return resp
			
		# Check if the user is looking for a suggestion
		#parsed = parser(sentence)
		#resp = self.look_for_description(parsed)
		#if resp is not None:
		#	return resp

		# If none of the above is correct

		# OTHER IDEA
		# if not resp:
		# 	(pronoun, noun, adjective, verb) = self.find_candidate_parts_of_speech(parsed)
		#	resp = self.comments_about_bot(pronoun, adjective)
		#	if not resp: resp = self.comments_about_self(pronoun, adjective)



	def match_description(self, sentence): # Without Parsing via Spacy
		descriptions = np.array(self.data.loc[:]['short_description'])
		vectors = self.data['vector']

		vector = seqWords2seqVec(sentence, self.word_index, self.embedding_matrix, self.max_num_vectors, self.num_features)

		sentencevec = self.encoder.predict(np.asarray([vector]))[0]
		indexes = self.findBestMatch(sentencevec, vectors)
		return indexes


	def findBestMatch(self, vector, descriptions):
		distances = np.zeros(len(descriptions))
		for i in range(0, len(descriptions)):
			distances[i] = eucDistance(vector, descriptions[i])
		indexes = distances.argsort()[:3]
		return indexes

	def suggest_city(self, indexes):
		resp = 'I would suggest {} cities:\n'.format(len(indexes))
		city_names = self.data['city_name'].tolist()
		city_descrp = self.data['short_description'].tolist()
		for index in indexes:
			resp += city_names[index] + ": " + city_descrp[index]
			resp += ". Read more about it in https://en.wikivoyage.org/wiki/{}".format(city_names[index])
			resp += "\n"
		return resp