import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
import pandas as pd

tree = ET.parse('data/data.xml')
root = tree.getroot()

citylist = dict()

counter = 0
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKMNOPQRSTUVWXYZ "
char_threshold = 15

re_title = re.compile(r'title')
re_revision = re.compile(r'revision')
re_text = re.compile(r'text')
re_cutter = re.compile(r'.*?==', re.DOTALL)

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
                        # Handle [[...]]
                        text_in_brackets = re.findall("\[\[[^\]]*\]\]", description)
                        for text in text_in_brackets:
                            if ".jpeg" in text or ".JPG" in text:
                                description = description.replace(text,'')
                                #description = re.sub(text, "",
                                #                     description)  # Remove everything between [[...]]
                            else:
                                description =  description.replace(text, text[2:len(text)-2])  # Remove [[...]]
                        # Handle {{...}}
                        description = re.sub("\{\{[^\}]*\}\}", "", description)  # Remove everything between {{...}}
                        # Handle the '
                        description = re.sub("'''", " ", description)
                        # Handle the \n
                        description = description.replace("\n", '')
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
            citylist[key] = description

        counter += 1
        if (counter >= 20):
            break;
# df = pd.DataFrame([[key,value] for key,value in citylist.iteritems()]) in python 2
df = pd.DataFrame([[key,value] for key,value in citylist.items()])
df.columns = ['city_name', "description"]
df.to_csv("proc_data/proc_data.csv")