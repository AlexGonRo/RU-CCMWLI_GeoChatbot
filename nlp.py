from __future__ import print_function, unicode_literals
import random
import logging
import numpy as np

# Import for DistanceMatrix
from datetime import datetime
import config
from spacy.en import English
import word_lists

key = 'AIzaSyDhjoU4gFGIJ5VTTaexR_4_tDoaMvbgCaA'
#####

# os.environ['NLTK_DATA'] = os.getcwd() + '/nltk_data'

# Set up spaCy


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
parser = English()


class NLP:
	def __init__(self, data):
		self.data = data

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

	def check_for_geo_keywords(self, sent):
		blindshot = False
		city1 = False
		city2 = False
		for token in sent:
			print(token.orth_, token.ent_type_)
			if token.ent_type_ == 'GPE':  # This is a verb
				if not city1: city1 = token.orth_
				if (token.orth_ != city1):
					city2 = token.orth_
					break
		if not city1:
			KEY_WORDS = (
				"where", "holidays", "holiday", "place", "sunny", "city", "town", "country", "destination", "warm",
				"nice", "sun", "beach", "sea", "ocean")
			for word in sent:
				if word.lower_ in KEY_WORDS: blindshot = True
		else:
			blindshot = False
		return city1, city2, blindshot


	def preprocess_text(self, sentence):
		"""Handle some weird edge cases in parsing, like 'i' needing to be capitalized
        to be correctly identified as a pronoun"""
		cleaned = []
		words = sentence.split(' ')
		for w in words:
			if w == 'i':
				w = 'I'
		if w == "i'm":
			w = "I'm"
		if w == "cya":
			w = "See you"
		cleaned.append(w)

		return ' '.join(cleaned)

	def find_pronoun(self, sent):
		"""Given a sentence, find a preferred pronoun to respond with. Returns None if no candidate
        pronoun is found in the input"""
		pronoun = None
		print("FIND_PRONOUN...")
		for token in sent:
			# Disambiguate pronouns
			if token.pos_ == 'PRON' and token.lower_ == 'you':
				pronoun = 'I'
			elif token.pos_ == 'PRON' and token.orth_ == 'I':
				# If the user mentioned themselves, then they will definitely be the pronoun
				pronoun = 'You'
			elif token.pos_ == 'PRON':
				pronoun = token.orth_
		return pronoun

	# end

	def find_verb(self, sent):
		# Pick a candidate verb for the sentence
		verb = None
		for token in sent:
			if token.pos_ == 'VERB':  # This is a verb
				verb = token.lemma_
		return verb

	def find_noun(self, sent):
		# Given a sentence, find the best candidate noun
		noun = None

		if not noun:
			for token in sent:
				if token.pos_ == 'NOUN':  # This is a noun
					noun = token.orth_
					break
		if noun:
			logger.info("Found noun: %s", noun)

		return noun

	def find_adjective(self, sent):
		"""Given a sentence, find the best candidate adjective."""
		adj = None
		for token in sent:
			if token.pos_ == 'ADJ':  # This is an adjective
				adj = token.orth_
				break
		return adj

	def comments_about_bot(self, pronoun, adjective):
		# if the input uses "YOU" than he talks about the bot.
		if (pronoun == "I" and adjective):
			resp = "I am indeed {adjective}".format(**{'adjective': adjective})
			return resp

	def comments_about_self(self, pronoun, adjective):
		# if the input uses "YOU" than he talks about the bot.
		if (pronoun == "You" and adjective):
			resp = "You aren't really {adjective}".format(**{'adjective': adjective})
			return resp


	def introduction(self):
		self.askorigin = True
		return word_lists.INTRODUCTION


	def city_in_sentence(self, sentence):
		city_names = self.data['city_name'].tolist()
		city_names_lower = [x.lower() for x in city_names]

		sentence_split = sentence.split()

		for word in sentence_split:
			if word in city_names_lower:
				return True, city_names[city_names_lower.index(word)]
		return False, None

	def describe_city(self, city):
		index = self.data.city_name[self.data.city_name == city].index.tolist()[0]
		description = self.data.iloc[index]['short_description']
		description = description + ". Read more about it in https://en.wikivoyage.org/wiki/{}".format(city)
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
		if city_in_sentence:
			resp = self.describe_city(city)
			return resp

		# Check if the user is looking for a suggestion
		parsed = parser(sentence)
		resp = self.look_for_description(parsed)
		if resp is not None:
			return resp

		# If none of the above is correct
		resp = self.random_answer()
		return resp

		# OTHER IDEA
		# if not resp:
		# 	(pronoun, noun, adjective, verb) = self.find_candidate_parts_of_speech(parsed)
		#	resp = self.comments_about_bot(pronoun, adjective)
		#	if not resp: resp = self.comments_about_self(pronoun, adjective)


	def find_candidate_parts_of_speech(self, parsed):
		# Given a parsed input, find the best pronoun, direct noun, adjective, and verb to match their input.
		# Returns a tuple of pronoun, noun, adjective, verb any of which may be None if there was no good match"""
		pronoun = None
		noun = None
		adjective = None
		verb = None
		print(parsed.sents)
		for sent in parsed.sents:
			print(sent)
			pronoun = self.find_pronoun(sent)
			noun = self.find_noun(sent)
			adjective = self.find_adjective(sent)
			verb = self.find_verb(sent)
		logger.info("Pronoun=%s, noun=%s, adjective=%s, verb=%s", pronoun, noun, adjective, verb)
		return pronoun, noun, adjective, verb

	def look_for_description(self,parsed_sentence):
		#TODO LOOK FOR TAG, NOT POS FOR ADJETIVES https://spacy.io/docs/usage/pos-tagging
		dict_noun_adjs = {}
		for token in parsed_sentence:
			if token.pos_ == 'ADJ':  # This is an adjective
				noun = token.head
				adj = token
				dict_noun_adjs[noun] = adj

		user_input = ""
		# In case we want to cut posesives refering to a person
		# for key, value in dict_noun_adjs.iteritems(): In python 2
		#for key, value in dict_noun_adjs.items():
		#	if key.pos_ == 'PRON' # This is a pronoun
		#       # DO NOT COUNT THIS ONE
		for key, value in dict_noun_adjs.items():
			user_input += key.text
			try:
				user_input += " ".join([word.text for word in value])
			except TypeError:
				user_input += " " + value.text


		tmp = self.data['vector'].tolist()
		index = self.findBestMatch(self.vectorize(user_input), tmp)

		return self.data.iloc[index]['city_name']


	def findBestMatch(self, vec1, vecarray):
		distances = np.zeros(len(vecarray))
		for i in range(0, len(vecarray)):
			distances[i] = self.eucDistance(vec1, vecarray[i])
		index = distances.argmin()
		return index

	def eucDistance(self, vector1, vector2):
		euclid = 0.0
		for (dim1, dim2) in zip(vector1, vector2):
			euclid = euclid + (dim1 - dim2) * (dim1 - dim2)
		# print dim1, dim2
		euclid = np.sqrt(euclid)
		return euclid

	def vectorize(self, word2vec, sentence, num_features=300):
		mywords = sentence.split(" ")
		myvector = np.zeros((num_features), dtype="float32")

		i = 0
		for word in mywords:
			# print word
			if word in word2vec.vocab:
				myvector = np.add(myvector, word2vec[word])  # Adding every new vector
				i += 1
		featureVec = np.divide(myvector, i)  # and in the end dividing it by the number of words

		return featureVec
	#def main():
#	nlp = NLP()
#	if (len(sys.argv) > 0):
#		saying = sys.argv[1]
#	else:
#		saying = "How are you, brobot?"
#	print(nlp.dback(saying))

#if __name__ == '__main__':
#	main()
