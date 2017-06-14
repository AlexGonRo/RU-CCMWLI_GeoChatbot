import numpy as np


def eucDistance(vector1, vector2):
    euclid = 0.0
    for (dim1, dim2) in zip(vector1, vector2):
        euclid = euclid +(dim1-dim2)*(dim1-dim2)
        #print dim1, dim2
    euclid = np.sqrt(euclid)
    return euclid


def maxDistance(vector1, vector2):
    return max(np.abs(np.substract(vector1,vector2)))


def minDistance(vec1, vec2):
    return min(np.abs(np.substract(vec1,vec2)))

def seqWords2seqVec(sentence, word_index, embedding_matrix, max_num_vectors, num_features):
    vector = []
    mywords = sentence.split(" ")
    count = 0
    # Substitute word by vector (if exists)
    for word in mywords:
        if word in word_index:
            vector.append(embedding_matrix[word_index[word]])
            count += 1
    # Zero-pad sentences or reduce size if needed
    if count < max_num_vectors:
        while True:
            tmp = [0 for i in range(num_features)]
            vector.append(tmp)
            if len(vector) == num_features:
                break
    elif count > max_num_vectors:
        vector = vector[:max_num_vectors]

    return vector