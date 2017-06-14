import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import os
import pickle
from seq2seq import Seq2seq
from utils import seqWords2seqVec

def preproc(data_path = 'data/data.xml'):


    # counter = 0 For testing porpuses

    # Array with all the information extracted from the data
    citylist = []
    # Valid characters
    char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKMNOPQRSTUVWXYZ "
    # Minimum length of the description (in characters)
    char_threshold = 100

    # Regular expressions
    re_title = re.compile(r'title')
    re_revision = re.compile(r'revision')
    re_cutter = re.compile(r'.*?==', re.DOTALL)
    re_spaces = re.compile(r'\s+')

    # Variables to navigate the xml file
    tree = ET.parse(data_path)
    root = tree.getroot()

    print("Starting to extract the descriptions of the cities.")
    count = 0
    for child in root:

        # Print run information
        count += 1
        if count % 5000 == 0:
            print('{} cities have been preprocessed.'.format(count))

        # Handle special case
        if (child.tag == "{http://www.mediawiki.org/xml/export-0.10/}siteinfo"):
            root.remove(child)
        else:
            # print 'CHILD'
            add_entry = True
            for subchild in child:
                if re_title.search(subchild.tag):
                    key = subchild.text
                    # Delete pages that are not articles
                    if key.startswith('Wikivoyage') or key.startswith('MediaWiki') or key.startswith('Template')\
                            or key.startswith('File') or key.startswith('Module') or key.startswith('Category')\
                            or key.startswith('Star articles'):
                        add_entry = False
                        break
                    # Delete cities which letters are not in the latin alphabet
                    if [char for char in key if char not in char_list]:
                        add_entry = False
                        break
                if re_revision.search(subchild.tag):
                    for text in subchild.findall('{http://www.mediawiki.org/xml/export-0.10/}text'):
                        description = text.text
                        if description is None or not description or len(description) > 0:
                            # Take just the description of the city
                            for section in re_cutter.finditer(description):  # Remove anything after first == appearance
                                description = section.group(0)
                                description = description[:-3]  # Remove also the == that were included in the regex
                                break;

                            # HANDLE #REDIRECT
                            if "#REDIRECT" in description:
                                add_entry = False
                                break

                            # Handle &nbsp;
                            description = re.sub('&nbsp;', ' ', description)

                            # Handle [[...]]
                            text_in_brackets = re.findall("\[\[[^\]]*\]\]", description)
                            for text in text_in_brackets:
                                if ".JPEG" in text or ".JPG" in text or ".jpeg" in text or ".jpg" in text :
                                    description = description.replace(text,'')
                                else:
                                    description =  description.replace(text, text[2:len(text)-2])  # Remove [[...]]

                            # Handle [...]
                            text_in_brackets = re.findall("\[[^\]]*\]", description)
                            if text_in_brackets:
                                for text in text_in_brackets:
                                    text_splitted = text.split(" ")
                                    description = description.replace(text_splitted[0], "")
                                    rest_text = " ".join(text_splitted[1:])
                                    description = description.replace(rest_text, rest_text[:len(rest_text)-1])

                            # Handle {{...}}
                            description = re.sub("\{\{[^\}]*\}\}", "", description)  # Remove everything between {{...}}

                            # Handle the \n
                            description = description.replace("\n", '')

                            # HANDLE SHORT DESCRIPTIONS
                            if len(description) < char_threshold:
                                add_entry = False
                                break

                            # HANDLE There are (at least) ...
                            if description.startswith('There are (at least)'):
                                add_entry = False
                                break

                            # Get first two sentences of text
                            short_description = get_short_description(description)
                            short_description = re_spaces.sub(' ', short_description) # Make sure that there is just
                                                                                        # one space between words.
                            if short_description[0] == ' ':
                                short_description = short_description[1:]

                            # Handle the '
                            description = re.sub("'''", " ", description)
                            # Everything goes to lowercase
                            description = description.lower()
                            # Remove stopwords
                            description_split = description.split(" ")
                            description = [word for word in description_split if word not in stopwords.words('english')]
                            description = " ".join(str(x) for x in description)
                            # Remove unwanted characters
                            description_char = [char for char in description if char in char_list]
                            description = "".join(str(x) for x in description_char)
                        else:
                            add_entry = False
                            break

            if add_entry:
                citylist.append([key, short_description, description])

            #counter += 1
            #if (counter >= 200):
            #    break;


    print("All the cities have been preprocessed.")

    print("Loading word embeddings...")
    word_index, embedding_matrix = get_vocabulary(citylist, 200)

    print("Calculating the description embeddings...")
    # Option 1
    # descriptions = [row[2] for row in citylist]
    # vector = [vectorize(word_index, embedding_matrix, description, 200) for description in descriptions]
    # Option 2
    descriptions = [row[2] for row in citylist]
    vector = vectorize_nn(word_index, embedding_matrix, descriptions, num_vectors = 200, num_features=200,
                          batch_size=32, latent_dim = 200, timesteps = 200, epochs = 5)

    print("Grouping all the information together...")
    citylist = [x + [vector[i]] for i, x in enumerate(citylist)]
    print("Saving data...")
    df = pd.DataFrame([[key, s_d, d, v] for key,s_d,d,v in citylist])
    df.columns = ['city_name', "short_description", "description", "vector"]
    df.to_pickle("proc_data/proc_data.pkl")
    print("The preprocesing is done.")


def get_short_description(description):
    description = re.sub("'''", " ", description)
    description = description.split('.')
    if len(description)>1:
        description = description[0] + " " + description [1]
    elif len(description)==1:
        description = description[0]
    else:
        description = None

    return description

def vectorize_nn(word_index, embedding_matrix, sentences, max_num_vectors = 200, num_features=200, batch_size=32,
                 latent_dim = 200, timesteps = 200, epochs = 5):
    print("Substituting words in descriptions by their vector representetion")
    vec_sentences = []
    sec_count = 0
    for i, sentence in enumerate(sentences):
        # Print run information
        sec_count += 1
        if sec_count % 5000 == 0:
            print('{} descriptions have been preprocessed.'.format(sec_count))

        vector = seqWords2seqVec(sentence, word_index, embedding_matrix, max_num_vectors, num_features)

        # Store vector
        vector = np.asarray(vector)
        vec_sentences.append(vector)

    print("All words have been subtituted by their vector representation")

    vec_sentences = np.asarray(vec_sentences)
    vec_sentences = np.reshape(vec_sentences,(len(sentences), num_vectors, num_features))
    # Create and train Neural net
    s2s = Seq2seq(num_vectors, latent_dim, timesteps, batch_size, word_index, embedding_matrix)
    print("Training autoencoder...")
    s2s.fit(vec_sentences,epochs)
    print("Getting vector representation of each description...")
    predictions = s2s.predict(vec_sentences)
    print("Saving neural network...")
    s2s.encoder.save('model/encoder.h5')
    return predictions



def vectorize(word_index, embedding_matrix, sentence, num_features=300):
    mywords = sentence.split(" ")
    myvector = np.zeros((num_features),dtype="float32")

    i = 0
    for word in mywords:
        if word in word_index:
            myvector = np.add(myvector,embedding_matrix[word_index[word]]) # Adding every new vector
            i+=1
    featureVec = np.divide(myvector, i) # and in the end dividing it by the number of words

    return featureVec

def get_vocabulary(citylist, vec_size = 200):

    short_descriptions = [row[1] for row in citylist]
    descriptions = [row[2] for row in citylist]

    total_text = short_descriptions + descriptions

    tokenizer = Tokenizer(num_words=55000)  # TODO: Check this hardcoded value. We could change it.
    tokenizer.fit_on_texts(total_text)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    with open("word_vectors/word_indx.csv", 'wb') as outfile:
        pickle.dump(word_index, outfile)

    # Load GloVe embeddings for all the words
    embeddings_index = {}
    f = open(os.path.join("GloVe", 'glove.6B.{}d.txt'.format(vec_size)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, vec_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    with open("word_vectors/embedding_matrix.csv", 'wb') as outfile:
        pickle.dump(embedding_matrix, outfile)

    return word_index, embedding_matrix


#preproc()