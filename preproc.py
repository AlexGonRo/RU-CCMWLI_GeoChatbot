import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
import pandas as pd



def preproc(data_path = 'data/data.xml'):
    tree = ET.parse(data_path)
    root = tree.getroot()

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
                                    #description = re.sub(text, "",
                                    #                     description)  # Remove everything between [[...]]
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
                        else:
                            add_entry = False
                            break

            if add_entry:
                citylist.append([key, short_description, description])

            counter += 1
            if (counter >= 20):
                break;

    df = pd.DataFrame([[key, s_d, d] for key,s_d,d in citylist])
    df.columns = ['city_name', "short_description", "description"]
    df.to_csv("proc_data/proc_data.csv")


def get_short_description(description):
    description = re.sub("'''", " ", description)
    description = description.split('.', 1)
    if len(description)>1:
        description = description[0] + " " + description [1]
    elif len(description)==1:
        description = description.split('.', 1)[0]
    else:
        description = None

    return description

preproc()