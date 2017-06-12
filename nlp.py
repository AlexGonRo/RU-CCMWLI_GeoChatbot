from __future__ import print_function, unicode_literals
import random
import logging
import numpy as np

# Import for DistanceMatrix
from datetime import datetime
import config
#from spacy.en import English
import word_lists
#Import for Match best City
from word2vec import *

key = 'AIzaSyDhjoU4gFGIJ5VTTaexR_4_tDoaMvbgCaA'
#####

# os.environ['NLTK_DATA'] = os.getcwd() + '/nltk_data'

# Set up spaCy


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#parser = English()


class NLP:
	def __init__(self, data, w2vmodel, descriptions_vec):
		self.data = data # THE PANDA DF
		self.w2vec = Word2Vec(w2vmodel, descriptions_vec)

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
			return self.match_description(sentence)
			
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
		df = self.data
		descriptions = np.array(self.data.loc[:]['description'])
		print("I am still good")
		sentencevec = self.w2vec.vectorize(sentence)
		index = self.w2vec.findBestMatch(sentencevec)
		return descriptions[index]

