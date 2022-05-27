from bs4 import BeautifulSoup
import os
from pathlib import Path
import json
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import random

class Preprocessor:
    def __init__(self):
        self.data = []
        self.dictionary = {}
        self.documents = {}
        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
        self.loadStopwords()

        self.train_data = {}
        self.test_data = {}

    def loadStopwords(self):
        self.stop_words = set(stopwords.words("english"))

    def tokenize(self, text):

        text = text.lower()  # case folding
        text = re.sub(r"-", " ", text)  # handling hyphen
        text = re.sub(r"[^\w\s]", " ", text)  # noise removal - replacing all types of [^a-zA-Z0-9] and [^\t\r\n\f] with space for splitting on space
        text = text.split()  # splitting on space
        return text

    # Lemmatization - WordNetLemmatizer used
    def Lemmatization(self, token):
        l = WordNetLemmatizer()
        return l.lemmatize(token)

    def remove_number_stopwords(self, tokens):

        filtered_sentence = []

        for w in tokens:
            if (w not in self.stop_words) and (not w.isnumeric()) and (len(w)!=1):
                filtered_sentence.append(self.Lemmatization(w))

        return filtered_sentence

    def PreprocessText(self, text):

        tokens = self.tokenize(text)
        tokens = self.remove_number_stopwords(tokens)
        return tokens

    def Scrape(self, folderNames):

        collectionDir = str(Path(__file__).parent.resolve()).replace("src", "course-cotrain-data")

        print(collectionDir)

        i = 0

        for folder in folderNames:

            temp = collectionDir + "\\" + folder

            print(temp)

            _class = temp.rpartition("\\")[-1]

            print(_class)

            if _class not in self.dictionary.keys():
                self.dictionary[_class] = {}

            for filename in os.listdir(temp):

                # if filename.endswith('.html') or filename.endswith('.edu'):

                fname = os.path.join(temp, filename)
                # print("Current file name ..", os.path.abspath(fname))

                with open(fname, "r") as file:

                    contents = file.read()

                    soup = BeautifulSoup(contents, "lxml")

                    for script in soup(["script", "style"]):
                        script.decompose()

                    text = list(soup.stripped_strings)
                    text = " ".join(text)

                    text = self.PreprocessText(text)

                    self.dictionary[_class][i] = filename
                    self.documents[i] = text

                    file.close()

                    i += 1

        # print(self.documents)
        self.WriteToDisk(self.dictionary, "dictionary")
        # self.WriteToDisk(self.documents, "documents")

    def split_train_test(self):

        self.total_docs = len(self.dictionary['course'].keys()) + len(self.dictionary['non-course'].keys())

        print(self.total_docs)

        train_split = round(self.total_docs * 0.80)
        test_split = round(self.total_docs * 0.20)

        print((train_split, test_split))

        course_test = round(test_split * 0.20)
        non_course_test = round(test_split * 0.80)

        print((course_test, non_course_test)) 

        test_doc_ids = {}
        test_doc_ids['course'] = random.sample(range(0, 229), course_test)
        test_doc_ids['non-course'] = random.sample(range(230, 1050), non_course_test)

        print(test_doc_ids)

        self.load_train_data(test_doc_ids)
        self.load_test_data(test_doc_ids)


    def load_train_data(self,test_doc_ids):

        temp = 0

        for _class in test_doc_ids.keys():

            for id in range(temp , temp + len(self.dictionary[_class].keys())):
                
                if id not in test_doc_ids[_class]:

                    self.train_data[id] = self.documents[id]

            temp = len(self.dictionary[_class].keys())

       
        self.WriteToDisk(self.train_data,"train_data")


    def load_test_data(self, test_doc_ids):
        
        for _class in test_doc_ids.keys():

            for id in test_doc_ids[_class]:
         
                self.test_data[id] = self.documents[id]

       
        self.WriteToDisk(self.test_data,"test_data")


    # writing specified index to disk
    def WriteToDisk(self, index, indexType):

        if not os.path.isdir(self.dataDir):
            os.mkdir(self.dataDir)

        filename = "\\" + indexType + ".txt"
        filePath = self.dataDir + filename

        with open(filePath, "w") as filehandle:
            filehandle.write(json.dumps(index))


# p = Preprocessor()
# s.Scrape(['fulltext\\course'])
# p.Scrape(['fulltext\\course','fulltext\\non-course','inlinks\\course','inlinks\\non-course'])
# p.Scrape(["fulltext\\course", "fulltext\\non-course"])
# p.split_train_test()
# s.Scrape('fulltext\\non-course')
# s.Scrape('inlinks\\non-course')
