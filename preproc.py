import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gzip
import pickle as cPickle
import numpy as np
import pandas as pd

def preproc(data_path = 'data/data.xml'):
    tree = ET.parse(data_path)
    root = tree.getroot()

    EMBEDDING_FILE = './model/GoogleNews-vectors-negative300.bin'
    # model = Word2Vec.load(EMBEDDING_FILE)

    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
                                                 binary=True)
    print("Loaded model")

    citylist = []

    counter = 0
    char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKMNOPQRSTUVWXYZ "
    char_threshold = 15

    re_title = re.compile(r'title')
    re_revision = re.compile(r'revision')
    re_text = re.compile(r'text')
    re_cutter = re.compile(r'.*?==', re.DOTALL)
    MULT_SPACES = re.compile(r'\s+')

    for child in root:
        # Handle special case
        if (child.tag == "{http://www.mediawiki.org/xml/export-0.10/}siteinfo"):
            root.remove(child)
        else:
            # print 'CHILD'
            add_entry = True
            for subchild in child:
                if re_title.search(subchild.tag):
                    # print "TITLEEEE"
                    # print subchild.text
                    key = subchild.text
                if re_revision.search(subchild.tag):
                    # print 'REVISION', subchild.tag, subchild.attrib
                    for text in subchild.findall('{http://www.mediawiki.org/xml/export-0.10/}text'):
                        # HANDLE #REDIRECT
                        if "#REDIRECT" in text.text:
                            add_entry = False
                            break
                        if len(text.text) > 0:
                            # Get just definition
                            description = text.text
                            for section in re_cutter.finditer(description):  # Remove anything after first == appearance
                                description = section.group(0)
                                description = description[:-3]  # Remove also the == that were included in the regex
                                break;
                            # HANDLE SHORT DESCRIPTIONS
                            if len(description) < char_threshold:
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

                            # Get first two sentences of text
                            short_description = get_short_description(description)
                            short_description = MULT_SPACES.sub(' ', short_description) # Make sure that there is just
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
                            # Remove stuff
                            # description = re.sub("[\\\{\}\[\]\\n]", "", description)
                            vector = vectorize(word2vec, description)
                        else:
                            add_entry = False
                            break

            if add_entry:
                citylist.append([key, short_description, description, vector])

            counter += 1
            if (counter >= 20):
                break;

    df = pd.DataFrame([[key, s_d, d] for key,s_d,d in citylist])
    df.columns = ['city_name', "short_description", "description", "vector"]
    df.to_csv("proc_data/proc_data.csv")


def get_short_description(description):
    description = re.sub("'''", " ", description)
    description = description.split('.')
    if len(description)>1:
        description = description[0] + " " + description [1]
    elif len(description)==1:
        description = description.split('.', 1)[0]
    else:
        description = None

    return description


def vectorize(word2vec, sentence, num_features=300):
    mywords = sentence.split(" ")
    myvector = np.zeros((num_features),dtype="float32")

    i = 0
    for word in mywords:
        #print word
        if word in word2vec.vocab:
            myvector = np.add(myvector,word2vec[word]) # Adding every new vector
            i+=1
    featureVec = np.divide(myvector, i) # and in the end dividing it by the number of words

    return featureVec

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



preproc()