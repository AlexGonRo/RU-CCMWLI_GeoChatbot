
from spacy.en import English

class Spacy(self, parser):
	
	def __init__(self):
		self.parser = English()

	

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


	def find_candidate_parts_of_speech(self, sent):
		sent = self.parser(sent)
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
