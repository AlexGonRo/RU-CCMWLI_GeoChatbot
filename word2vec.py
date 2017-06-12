from gensim.models import KeyedVectors
import cPickle, gzip
import numpy as np
import pandas as pd
import gzip, pickle as cPickle

class Word2Vec:
    
    def __init__(self,w2vmodel, descriptions_vec):
	self.word2vec = w2vmodel
	self.descriptions_vec = descriptions_vec

    def vectorizeArray(self,descriptions):
        return [vectorize(description) for description in descriptions]

    def vectorize(self, sentence, num_features=300):
        mywords = sentence.split(" ")
        myvector = np.zeros((num_features),dtype="float32")

        i = 0
        for word in mywords:
            #print word
            if word in self.word2vec.vocab:
                myvector = np.add(myvector,self.word2vec[word]) # Adding every new vector
                i+=1
        featureVec = np.divide(myvector, i) # and in the end dividing it by the number of words

        return featureVec

    def eucDistance(self, vector1, vector2):
        euclid = 0.0
        for (dim1, dim2) in zip(vector1, vector2):
            euclid = euclid +(dim1-dim2)*(dim1-dim2)
            #print dim1, dim2
        euclid = np.sqrt(euclid)
        return euclid
    def maxDistance(self, vector1, vector2):
        return max(np.abs(np.substract(vector1,vector2)))
    def minDistance(self, vec1, vec2):
        return min(self,np.abs(np.substract(vec1,vec2)))
    def findBestMatch(self, vector):
        distances = np.zeros(len(self.descriptions_vec))
        for i in range(0,len(self.descriptions_vec)):
            distances[i] = self.eucDistance(vector, self.descriptions_vec[i])
        index = distances.argmin()
        return index             

    def save_zipped_pickle(self, obj, filename, protocol=-1):
        with gzip.open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol)
            f.close()

    def load_zipped_pickle(self, filename):	# loads and unpacks
        with gzip.open(filename, 'rb') as f:
            loaded_object = cPickle.load(f)
            f.close()
            return loaded_object
