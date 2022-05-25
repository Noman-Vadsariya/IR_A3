from bs4 import BeautifulSoup
import os
from pathlib import Path
import json
import re
from matplotlib.pyplot import cla
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:

    def __init__(self):
        self.data = []
        self.dictionary = {}
        self.documents = {}
        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
        self.loadStopwords()

    def loadStopwords(self):
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self,text):

        text = text.lower()                   # case folding
        text = re.sub(r"-", " ", text)        # handling hyphen
        text = re.sub(r"[^\w\s]", " ", text)  # noise removal - replacing all types of [^a-zA-Z0-9] and [^\t\r\n\f] with space for splitting on space
        text = text.split()                   # splitting on space
        return text

    def remove_number_stopwords(self,tokens):
        
        filtered_sentence = []

        for w in tokens:
            if (w not in self.stop_words) and w.isalpha():
                filtered_sentence.append(self.Lemmatization(w))

        return filtered_sentence
    

    # Lemmatization - WordNetLemmatizer used
    def Lemmatization(self, token):
        l = WordNetLemmatizer()
        return l.lemmatize(token)


    def PreprocessText(self,text):

        tokens = self.tokenize(text)
        tokens = self.remove_number_stopwords(tokens)
        return tokens
        
        
    def Scrape(self,folderNames):

        collectionDir = str(Path(__file__).parent.resolve()).replace("src", "course-cotrain-data")

        print(collectionDir)
        
        i = 0

        for folder in folderNames:

            temp = collectionDir + '\\' + folder

            print(temp)

            _class = temp.rpartition('\\')[-1]

            print(_class)

            self.dictionary[_class] = {}

            for filename in os.listdir(temp):
                
                # if filename.endswith('.html') or filename.endswith('.edu'):
                    
                fname = os.path.join(temp, filename)
                # print("Current file name ..", os.path.abspath(fname))
                
                with open(fname, 'r') as file:
                    
                    contents = file.read()

                    soup = BeautifulSoup(contents, 'lxml')

                    for script in soup(["script", "style"]):
                        script.decompose()

                    text = list(soup.stripped_strings)
                    text = ' '.join(text)
                    
                    text = self.PreprocessText(text)

                    self.dictionary[_class][i] = filename
                    self.documents[i] = text

                    file.close()
            
                    i+=1

        # print(self.documents)
        self.WriteToDisk(self.dictionary,'dictionary')
        self.WriteToDisk(self.documents,'documents')


    # writing specified index to disk
    def WriteToDisk(self, index, indexType):

        if not os.path.isdir(self.dataDir):
            os.mkdir(self.dataDir)

        filename = "\\" + indexType + ".txt"
        filePath = self.dataDir + filename

        with open(filePath, "w") as filehandle:
            filehandle.write(json.dumps(index))


p = Preprocessor()

# s.Scrape(['fulltext\\course'])
# s.Scrape(['fulltext\\course','fulltext\\non-course','inlinks\\course','inlinks\\non-course'])
p.Scrape(['fulltext\\course','fulltext\\non-course'])
# s.Scrape('fulltext\\non-course')
# s.Scrape('inlinks\\non-course')
